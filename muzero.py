# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test
from typing import Any

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
from config import MuZeroConfig
from games.game import ReplayBuffer, Game, make_atari_config
from mcts import Node, expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from storage import SharedStorage
from utils import MinMaxStats


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    #while True:
    for _ in range(config.num_episodes):
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
        backpropagate([root], network_output.value, game.to_play(), config.discount,
                      min_max_stats)
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


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
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

            # TODO: Freeze here, because target_policy return a empty list (terminal states?)

            l = tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.stack(list(network_output.policy_logits.values())),
                labels=tf.convert_to_tensor(target_policy))
            l += scalar_loss(network_output.value, target_value)
            if k > 0:
                l += scalar_loss(network_output.reward, tf.constant(target_reward, shape=(1, 1)))

            loss += scale_gradient(l, gradient_scale)
    loss /= len(batch)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss, var_list=network.get_variables())
    network.increment_training_steps()


def scalar_loss(prediction: tf.Tensor, target: tf.Tensor) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.

    if isinstance(prediction, float):
        prediction = tf.constant(prediction, shape=(1, 1))

    if isinstance(target, float):
        target = tf.constant(target, shape=(1, 1))

    return CategoricalCrossentropy()(target, prediction).numpy()


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

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


if __name__ == "__main__":
    config = make_atari_config()
    muzero(config)
