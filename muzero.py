# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test
from pathlib import Path
from threading import Thread
from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
# from tqdm import trange
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tqdm.autonotebook import trange

from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from storage import SharedStorage
from summary import write_summary_loss, write_summary_score
from utils import MinMaxStats, scalar_to_support


##################################
####### Part 1: Self-Play ########

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = Game(config.action_space_size, config.discount)

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        network_output = network.initial_inference(current_observation)
        expand_node(root, game.to_play(), game.legal_actions(), network_output)
        backpropagate([root], network_output.value, game.to_play(), config.discount, min_max_stats)
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network, min_max_stats)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: MuZeroConfig,
                  storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    #lr_schedule = ExponentialDecay(initial_learning_rate=config.lr_init, decay_steps=10, decay_rate=0.9)
    #optimizer = Adam(learning_rate=lr_schedule)
    optimizer = Adam(learning_rate=0.05)

    for i in range(config.training_steps):

        if i % config.checkpoint_interval == 0:
            storage.save_network(network)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        loss = update_weights(optimizer, network, batch, config.weight_decay)

        write_summary_loss(float(loss), replay_buffer.loss_counter)
        replay_buffer.loss_counter += 1

        save_checkpoints(storage.latest_network())

    storage.save_network(network)


def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    loss = 0

    with tf.GradientTape() as f_tape, tf.GradientTape() as g_tape, tf.GradientTape() as h_tape:
        for observations, actions, targets in batch:
            # Initial step, from the real observation.
            network_output = network.initial_inference(observations)
            hidden_state = network_output.hidden_state
            predictions = [(1.0, network_output)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                network_output = network.recurrent_inference(hidden_state, action)
                hidden_state = network_output.hidden_state
                predictions.append((1.0 / len(actions), network_output))

                hidden_state = scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):
                gradient_scale, network_output = prediction
                target_value, target_reward, target_policy = target

                if target_policy:
                    p_logits = tf.stack(list(network_output.policy_logits.values()))
                    p_labels = tf.convert_to_tensor(target_policy)
                    policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits, labels=p_labels)

                    value_loss = scalar_loss(network_output.value, target_value)
                    reward_loss = 0.0

                    if k > 0:
                        reward_loss = scalar_loss(network_output.reward, target_reward)

                    loss += scale_gradient((value_loss * 0.25) + policy_loss + reward_loss, gradient_scale)

        loss /= len(batch)

        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)

    f_grad = f_tape.gradient(loss, network.f_prediction.trainable_variables)
    g_grad = g_tape.gradient(loss, network.g_dynamics.trainable_variables)
    h_grad = h_tape.gradient(loss, network.h_representation.trainable_variables)
    optimizer.apply_gradients(zip(f_grad, network.f_prediction.trainable_variables))
    optimizer.apply_gradients(zip(g_grad, network.g_dynamics.trainable_variables))
    optimizer.apply_gradients(zip(h_grad, network.h_representation.trainable_variables))

    network.increment_training_steps()
    return loss


def scalar_loss(prediction, target):
    target = tf.constant(target, dtype=tf.float32, shape=(1,))
    prediction = tf.constant(prediction, dtype=tf.float32, shape=(1,))

    target = scalar_to_support(np.array([float(target)]), 20)
    prediction = scalar_to_support(np.array([float(prediction)]), 20)
    return tf.cast(tf.losses.categorical_crossentropy(target, prediction), dtype=tf.float32)
    # target = tf.math.sign(target) * (tf.math.sqrt(tf.math.abs(target) + 1) - 1) + 0.001 * target
    # return tf.reduce_sum(tf.keras.losses.MSE(y_true=target, y_pred=prediction))
    # return tf.reduce_sum(-target * tf.math.log(prediction))
    # return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(target, prediction))


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################
def launch_job(f, *args):
    f(*args)


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    with trange(5000) as t:
        for i in range(5000):
            run_selfplay(config, storage, replay_buffer)
            train_network(config, storage, replay_buffer)
            export_models(storage.latest_network())
            score_mean = tf.reduce_mean([tf.reduce_sum(game.rewards) for game in replay_buffer.buffer])
            t.set_description(f"Score Mean: {score_mean:.2f}")
            t.update(1)
            t.refresh()


def save_checkpoints(network: Network):
    try:
        for model in network.get_networks():
            Path.mkdir(Path(f'./checkpoints/{model.__class__.__name__}'), parents=True, exist_ok=True)
            model.save_weights(f'./checkpoints/{model.__class__.__name__}/checkpoint')
    except Exception as e:
        print(f"Unable to save networks. {e}")


def load_checkpoints(network: Network):
    try:
        for model in network.get_networks():
            path = Path(f'./checkpoints/{model.__class__.__name__}/checkpoint')
            if Path.exists(path):
                model.load_weights(path)
                print(f"Load weights with success.")
    except Exception as e:
        print(f"Unable to load networks. {e}")


def export_models(network: Network):
    try:
        for model in network.get_networks():
            path = Path(f'./data/saved_model/{model.__class__.__name__}')
            Path.mkdir(path, parents=True, exist_ok=True)
            tf.saved_model.save(model, str(path.absolute()))
            #print(f"Model {model.__class__.__name__} Saved!")
    except Exception as e:
        print(f"Unable to save networks. {e}")


if __name__ == "__main__":
    config = make_atari_config()
    with tf.device('/device:GPU:0'):
        muzero(config)
