import ray
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from buffers import ReplayBuffer
from config import MuZeroConfig
from models.network import Network, scale_gradient, scalar_loss
from stats import Summary
from storage import SharedStorage


@ray.remote
class Leaner:
    def __init__(self, config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, summary: Summary):
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.summary = summary
        self.network = Network(self.config)
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=self.config.lr_init,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate
        )
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    def start(self):
        count = 0
        while self.network.training_steps() < 10000:
            size = ray.get(self.replay_buffer.size.remote())
            if size > 2:
                count += 1
                # temperature = visit_softmax_temperature(self.network.training_steps(), self.config)
                if count % self.config.checkpoint_interval == 0:
                    weigths = self.network.get_weights()
                    self.storage.update_network.remote(weigths)

                self.update_weights()

    def update_weights(self):
        batch = ray.get(self.replay_buffer.sample_batch.remote())

        with GradientTape() as tape:
            loss = self.network.loss_function(batch)

        grads = tape.gradient(loss, self.network.get_variables())
        self.optimizer.apply_gradients(zip(grads, self.network.get_variables()))
        self.summary.loss.remote(loss)
        self.summary.publish_loss.remote(self.network.training_steps())
        self.network.update_training_steps()
