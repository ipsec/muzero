# -*- coding: utf-8 -*-
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tqdm import trange

from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from storage import SharedStorage
from utils import MinMaxStats, tf_scalar_to_support, tf_scalar_to_support_batch
from utils.exports import export_models


def write_summary_score_current(step):
    with summary_writer.as_default():
        tf.summary.scalar("Score Current", train_score_current.result(), step)
        train_score_current.reset_states()


def write_summary_score(step):
    with summary_writer.as_default():
        tf.summary.scalar("Score Mean", train_score_mean.result(), step)


def write_summary_loss(step):
    with summary_writer.as_default():
        tf.summary.scalar("Loss Mean", train_loss.result(), step)


def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    # while True:
    network = storage.latest_network()
    game = play_game(config, network)
    train_score_mean(tf.reduce_sum(game.rewards))
    train_score_current(tf.reduce_sum(game.rewards))
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = Game(config.discount)

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
        action = select_action(root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


def train_network(config: MuZeroConfig,
                  storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network(config)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=config.lr_init,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate
    )
    optimizer = Adam(learning_rate=lr_schedule)
    count = 0
    mean_count = 0

    t = trange(config.training_steps, desc='Training', leave=True)
    for i in t:

        desc = f"Games (played): {len(replay_buffer.buffer)} - " \
               f"Score Mean: {float(train_score_mean.result()):.2f}"

        t.set_description(desc)
        t.refresh()

        if i % config.checkpoint_interval == 0:
            g = trange(config.num_actors, desc='Playing', leave=False)
            for _ in g:
                run_selfplay(config, storage, replay_buffer)
                train_score_current(tf.reduce_sum(replay_buffer.buffer[-1].rewards))
                train_score_mean(tf.reduce_sum(replay_buffer.buffer[-1].rewards))
                write_summary_score_current(count)
                count += 1

            write_summary_score(mean_count)
            mean_count += 1

            storage.save_network(i, network)
            # network.save_checkpoint()

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        loss = update_weights(optimizer, network, batch, config.weight_decay)
        train_loss(loss)
        write_summary_loss(i)

    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss_batch(prediction, target):
    prediction = tf.cast(prediction, dtype=tf.float64)
    prediction = tf.experimental.numpy.atleast_2d(prediction)

    target = tf.cast(target, dtype=tf.float64)
    target = tf.experimental.numpy.atleast_2d(target)

    target = tf_scalar_to_support_batch(target, 20)
    prediction = tf_scalar_to_support_batch(prediction, 20)

    return tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target), dtype=tf.float32)


def scalar_loss(prediction, target, with_support: bool = False):
    target = tf.cast([[target]], dtype=tf.float32)
    prediction = tf.cast([[prediction]], dtype=tf.float32)

    if with_support:
        target = tf_scalar_to_support(target, 20)
        prediction = tf_scalar_to_support(prediction, 20)
        return tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target)

    # Without Support use only MSE
    return tf.reduce_sum(tf.losses.MSE(target, prediction))


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

            if target_reward != 1.0 and target_reward != 0.0:
                print(f"value: {target_value}")
                print(f"reward: {target_reward}")
                print(f"policy: {target_policy}")
                print("##########################")

            policy_loss = 0.

            if target_policy:
                policy_logits = tf.stack(list(network_output.policy_logits.values()))
                policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=target_policy)

            value_loss = scalar_loss(network_output.value, target_value, with_support=False)

            reward_loss = 0.

            if k > 0:
                reward_loss += scalar_loss(network_output.reward, target_reward, with_support=False)

            local_loss = tf.reduce_sum(policy_loss + value_loss + reward_loss)

            loss += scale_gradient(local_loss, gradient_scale)

    loss /= len(batch)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    return loss


def update_weights(optimizer: tf.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    with tf.GradientTape() as tape:
        loss = compute_loss(network, batch, weight_decay)

    train_loss(loss)

    grads = tape.gradient(loss, network.get_variables())
    optimizer.apply_gradients(zip(grads, network.get_variables()))

    network.increment_training_steps()
    return loss


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    train_network(config, storage, replay_buffer)
    export_models(storage.latest_network())


if __name__ == "__main__":
    np.random.seed(12345)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_score_mean = tf.keras.metrics.Mean('train_score_mean', dtype=tf.float32)
    train_score_current = tf.keras.metrics.Mean('train_score_current', dtype=tf.float32)

    data_path = Path("./data/muzero")
    data_path.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")

    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    muzero(make_atari_config(env))
