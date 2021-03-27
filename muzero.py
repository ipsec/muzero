# -*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf

from tqdm import trange
from pathlib import Path
from threading import Thread
from multiprocessing import Pool, cpu_count

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from storage import SharedStorage
from utils import MinMaxStats, tf_scalar_to_support
from utils.exports import export_models


def write_summary_score(step):
    with summary_writer.as_default():
        tf.summary.scalar("Score Current", train_score_current.result(), step)
        tf.summary.scalar("Score Mean", train_score_mean.result(), step)
        tf.summary.scalar("Loss Mean", train_loss.result(), step)
        train_score_current.reset_states()


def run_selfplay(config: MuZeroConfig, replay_buffer: ReplayBuffer):
    # with Pool(int(np.ceil(cpu_count() / 2))) as pool:
    with Pool(1) as pool:
        while True:
            counter = len(replay_buffer.buffer) + len(replay_buffer.buffer_tmp)
            results = [pool.apply_async(func=play_game, args=(config, counter)) for _ in range(config.num_actors)]
            for p in results:
                game = p.get()
                replay_buffer.save_game(game)
                train_score_mean(tf.reduce_sum(game.rewards))
                train_score_current(tf.reduce_sum(game.rewards))

    # game = play_game(config, network)
    # train_score_mean(tf.reduce_sum(game.rewards))
    # train_score_current(tf.reduce_sum(game.rewards))
    # replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, counter: int) -> Game:
    game = Game(config.discount)
    network = Network(config)
    network.restore()
    network.training_steps_set(counter)

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

    t = trange(config.training_steps, desc='Training', leave=True)
    for i in t:

        desc = f"Games (training/played): " \
               f"{len(replay_buffer.buffer_tmp)}/{len(replay_buffer.buffer_tmp)} - " \
               f"Score Mean: {float(train_score_mean.result()):.2f}"

        if i > 0:
            desc = f"Games (training/played): " \
                   f"{len(replay_buffer.buffer)}/{len(replay_buffer.buffer_tmp)} - " \
                   f"Score Mean: {float(train_score_mean.result()):.2f}"

        t.set_description(desc)
        t.refresh()

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        loss = update_weights(optimizer, network, batch, config.weight_decay)
        train_loss(loss)
        write_summary_score(i)

        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
            network.save_checkpoint()
            replay_buffer.update_main()

    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss(prediction, target):
    prediction = tf.cast(prediction, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    target = tf.experimental.numpy.atleast_2d(target)
    prediction = tf.experimental.numpy.atleast_2d(prediction)

    target = tf_scalar_to_support(target, 20)
    prediction = tf_scalar_to_support(prediction, 20)

    res = tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target), dtype=tf.float32)
    return res


def scalar_loss_mse(y_true, y_pred):
    y_true = tf.constant(y_true, dtype=tf.float32, shape=(1, 1))
    y_pred = tf.constant(y_pred, dtype=tf.float32, shape=(1, 1))
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)


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

            local_loss = 0.

            if target_policy:  # How to treat absorbing states? Just pass?
                policy_logits = tf.stack(list(network_output.policy_logits.values()))
                target_policy = tf.convert_to_tensor(target_policy)
                local_loss = tf.nn.softmax_cross_entropy_with_logits(target_policy, policy_logits)

            # local_loss += scalar_loss(network_output.value, target_value)
            local_loss += scalar_loss_mse(target_value, network_output.value)

            if k > 0:
                # local_loss += scalar_loss(network_output.reward, target_reward)
                local_loss += scalar_loss_mse(target_reward, network_output.reward)

            loss += scale_gradient(local_loss, gradient_scale)
    loss /= len(batch)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    return loss


def update_weights(optimizer: tf.optimizers.Optimizer, network: Network, batch, weight_decay: float):
    with tf.GradientTape() as tape:
        loss = compute_loss(network, batch, weight_decay)

    train_loss(loss)
    variables = network.get_variables()
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    network.increment_training_steps()
    return loss


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    thread_games = Thread(target=run_selfplay, args=(config, replay_buffer))
    thread_games.start()

    while len(replay_buffer.buffer_tmp) < 1:
        pass

    replay_buffer.update_main()

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
    muzero(make_atari_config(env))
