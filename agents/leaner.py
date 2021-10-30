import ray
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from buffers import ReplayBuffer
from config import MuZeroConfig
from models.network import Network
from stats import create_summary
from storage import SharedStorage


@ray.remote
class Leaner:
    def __init__(self, config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.summary = create_summary(name="leaner")
        self.metrics_loss = Mean(f'leaner-loss', dtype=tf.float32)
        self.network = Network(self.config)
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=self.config.lr_init,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate
        )
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    def start(self):
        while self.network.training_steps() < self.config.training_steps:
            if ray.get(self.replay_buffer.size.remote()) > 0:

                self.train()

                if self.network.training_steps() % self.config.checkpoint_interval == 0:
                    weigths = self.network.get_weights()
                    self.storage.update_network.remote(weigths)

                if self.network.training_steps() % self.config.save_interval == 0:
                    self.network.save()

        print("Finished")

    def train(self):
        batch = ray.get(self.replay_buffer.sample_batch.remote())

        with tf.GradientTape() as tape:
            loss = self.network.loss_function(batch)

        grads = tape.gradient(loss, self.network.get_variables())
        self.optimizer.apply_gradients(zip(grads, self.network.get_variables()))

        self.metrics_loss(loss)
        with self.summary.as_default():
            tf.summary.scalar(f'loss', self.metrics_loss.result(), self.network.training_steps())
        self.metrics_loss.reset_states()

        self.network.update_training_steps()
