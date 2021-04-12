# -*- coding: utf-8 -*-
from pathlib import Path
from threading import Thread

import ray
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tqdm import trange

from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action, \
    visit_softmax_temperature
from models.network import Network
from storage import SharedStorage
from utils import MinMaxStats, tf_scalar_to_support
from utils.exports import export_models


def write_game_count(step):
    with summary_writer.as_default():
        tf.summary.scalar("Games count", games_count.result(), step)


def writer_score_summary(step: int, tag: str, value: float):
    with summary_writer.as_default(step=step):
        tf.summary.scalar(tag, value, step)


def write_summary_loss(step):
    with summary_writer.as_default():
        tf.summary.scalar("Loss", train_loss.result(), step)
        train_loss.reset_states()


def run_selfplay(config: MuZeroConfig, replay_buffer: ReplayBuffer, tag: str):
    count = 0
    while True:
        game = ray.get(play_game.remote(config))
        replay_buffer.save_game(game)
        train_score_mean(sum(game.rewards))
        writer_score_summary(count, tag, sum(game.rewards))
        games_count(1)
        count += 1


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
@ray.remote
def play_game(config: MuZeroConfig) -> Game:
    game = Game(config.discount)
    network = Network(config)
    network.restore()

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.get_observation_from_index(-1)
        network_output = network.initial_inference(current_observation)
        expand_node(root, game.to_play(), game.legal_actions(), network_output)
        backpropagate([root], network_output.value, game.to_play(), config.discount, min_max_stats)
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network, min_max_stats)
        action = select_action(root, network, config)
        game.apply(action)
        game.store_search_statistics(root)

    return game


def train_network(config: MuZeroConfig,
                  storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    storage.save_network(0, network)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=config.lr_init,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate
    )

    optimizer = Adam(learning_rate=lr_schedule)

    with trange(config.training_steps) as t:
        updated = 0
        for i in t:
            t.set_postfix(
                temperature=visit_softmax_temperature(network.training_steps_counter(), config),
                updated=updated
            )

            if i % config.checkpoint_interval == 0:
                replay_buffer.update()
                storage.save_network(i, network)
                updated = i
                network.save(i)
                network.save_latest()

            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)

            update_weights(optimizer, network, batch, config.weight_decay)
            write_summary_loss(i)
            write_game_count(i)

    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def scalar_loss(target, prediction, with_support: bool = False):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)

    target = tf.reshape(target, [1, 1])
    prediction = tf.reshape(prediction, [1, 1])

    if with_support:

        target = tf_scalar_to_support(target, 20)
        prediction = tf_scalar_to_support(prediction, 20)

    return tf.losses.MSE(target, prediction)


def update_weights(optimizer: tf.optimizers.Optimizer, network: Network, batch, weight_decay: float):

    def compute_loss():
        loss = 0.0
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            network_output = network.initial_inference(image, training=True)
            hidden_state = network_output.hidden_state
            predictions = [(1.0, network_output)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                network_output = network.recurrent_inference(hidden_state, action, training=True)
                hidden_state = network_output.hidden_state
                predictions.append((1.0 / len(actions), network_output))
                hidden_state = scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):
                gradient_scale, network_output = prediction
                target_value, target_reward, target_policy = target

                # Policy Loss
                if target_policy:
                    l = 0.0
                    
                    policy_logits = tf.stack(list(network_output.policy_logits.values()))
                    target_policy = tf.convert_to_tensor(target_policy)
                    l = tf.nn.softmax_cross_entropy_with_logits(labels=target_policy, logits=policy_logits)

                    # Value loss
                    l += scalar_loss(target_value, network_output.value, with_support=True) * 0.25

                    if k > 0:
                        l += scalar_loss(target_reward, network_output.reward, with_support=True)

                    loss += scale_gradient(l, gradient_scale)

        loss /= len(batch)

        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)

        train_loss(loss)

        return loss

    optimizer.minimize(loss=compute_loss, var_list=network.get_variables())
    network.increment_training_steps()


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_actors):
        thread = Thread(target=run_selfplay, args=(config, replay_buffer, f'Game-{i}'))
        thread.start()

    while len(replay_buffer.buffer_tmp) < config.batch_size:
        pass

    train_network(config, storage, replay_buffer)
    export_models(storage.latest_network())


if __name__ == "__main__":

    ray.init()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_score_mean = tf.keras.metrics.Mean('train_score_mean', dtype=tf.float32)
    games_count = tf.keras.metrics.Sum('games_total', dtype=tf.float32)

    data_path = Path("./data/muzero")
    data_path.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")

    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    muzero(make_atari_config(env))
