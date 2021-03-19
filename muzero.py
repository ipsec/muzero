# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test
import logging
from threading import Thread

import gym
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
from tqdm import trange

from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from storage import SharedStorage
from summary import write_summary_loss
from utils import MinMaxStats, tf_scalar_to_support
from utils.exports import save_checkpoints, export_models

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(threadName)s %(message)s')


##################################
####### Part 1: Self-Play ########

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    while True:
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
    lr_schedule = ExponentialDecay(
        initial_learning_rate=config.lr_init,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate
    )
    # optimizer = Adam(learning_rate=lr_schedule)
    # optimizer = Adam(learning_rate=config.lr_init)
    # optimizer = SGD(learning_rate=config.lr_init, momentum=config.momentum)
    # optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    # optimizer = SGD(learning_rate=0.0001, momentum=0.1)
    optimizer = Adam(learning_rate=0.0001)

    with trange(config.training_steps) as t:
        for i in t:
            reward_mean = np.mean([np.sum(game.rewards) for game in replay_buffer.buffer])
            msg = f"Games: {len(replay_buffer.buffer):05d} - Reward mean: {reward_mean:05.2f}"

            t.set_description(msg)
            t.update(1)
            t.refresh()

            if i % config.checkpoint_interval == 0:
                storage.save_network(i, network)
                save_checkpoints(storage.latest_network())
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            loss = update_weights(optimizer, network, batch, config.weight_decay)
            write_summary_loss(float(loss), network.training_steps_counter())
            # logging.debug(f"Step: {network.training_steps_counter():05d} - Loss: {float(loss):.2f}")

    storage.save_network(config.training_steps, network)


# @tf.function
def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


# @tf.function
def scalar_loss(prediction, target):
    target = np.atleast_2d(target)
    prediction = np.atleast_2d(prediction)

    target = tf_scalar_to_support(target, 300)
    prediction = tf_scalar_to_support(prediction, 300)

    res = tf.cast(tf.reduce_sum(-target * tf.nn.log_softmax(prediction)), dtype=tf.float32)
    return res


# @tf.function
def compute_loss(network: Network, batch, weight_decay: float):
    loss = 0
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        network_output = network.initial_inference(image)
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

            if not target_policy:  # How to treat absorbing states? Just pass?
                continue

            l = tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.stack(list(network_output.policy_logits.values())), labels=target_policy)

            l += scalar_loss(network_output.value, target_value)
            if k > 0:
                l += scalar_loss(network_output.reward, target_reward)

            loss += scale_gradient(l, gradient_scale)
    loss /= len(batch)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    return loss


def get_variables(network):
    parts = (network.f_prediction, network.g_dynamics, network.h_representation)
    return [v for v_list in map(lambda n: n.trainable_weights, parts) for v in v_list]


def update_weights(optimizer: tf.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    with tf.GradientTape() as tape:
        loss = compute_loss(network, batch, weight_decay)

    grads = tape.gradient(loss, get_variables(network))
    optimizer.apply_gradients(zip(grads, get_variables(network)))

    network.increment_training_steps()
    return loss


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

    thread = Thread(target=run_selfplay, args=(config, storage, replay_buffer))
    thread.start()

    while len(replay_buffer.buffer) == 0:
        pass

    train_network(config, storage, replay_buffer)
    export_models(storage.latest_network())


if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        env = gym.make('CartPole-v1')
        config = make_atari_config(env)
        muzero(config)
