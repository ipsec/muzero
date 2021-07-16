# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

import collections
import math
import traceback
import typing
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from icecream import ic


##########################
####### Helpers ##########


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -float('inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

def scale(t: tf.Tensor):
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))

def scalar_to_support(x: tf.Tensor, support_size: int, eps: float = 0.001) -> tf.Tensor:
    if support_size == 0:
        return x

    x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + eps * x

    transformed = tf.clip_by_value(x, -support_size, support_size - 1e-6)
    floored = tf.floor(transformed)
    prob = transformed - floored

    idx_0 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size), dtype=tf.int32), -1)
    idx_1 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size + 1), dtype=tf.int32), -1)
    idx_0 = tf.stack([tf.range(x.shape[1]), idx_0])
    idx_1 = tf.stack([tf.range(x.shape[1]), idx_1])
    indexes = tf.squeeze(tf.stack([idx_0, idx_1]))

    updates = tf.squeeze(tf.concat([1 - prob, prob], axis=0))
    return tf.scatter_nd(indexes, updates, (1, 2 * support_size + 1))

def support_to_scalar(x: tf.Tensor, support_size: int, eps: float = 0.001) -> tf.Tensor:
    if support_size == 0:
        return x

    x = tf.nn.softmax(x)
    bins = tf.range(-support_size, support_size + 1, dtype=tf.float32)
    v = tf.tensordot(tf.squeeze(x), tf.squeeze(bins), 1)

    value = tf.math.sign(v) * (((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(v) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
    return tf.expand_dims([value], axis=0)

class MuZeroConfig(object):

    def __init__(self,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 known_bounds: Optional[KnownBounds] = None):

        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = 10
        self.window_size = 200
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        # Game
        # self.env = gym.make('LunarLander-v2')
        self.env = gym.make('CartPole-v1')

        ### Self-Play
        self.action_space_size = self.env.action_space.n
        self.num_actors = num_actors

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_config() -> MuZeroConfig:
    return MuZeroConfig(
        max_moves=500,
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=10,
        batch_size=32,
        td_steps=10,
        num_actors=8,
        lr_init=0.05,
        lr_decay_steps=350e3,
        #known_bounds=KnownBounds(0, 1),
    )


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class Player(object):
    pass


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory(object):
    """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return list(range(self.action_space_size))

    def to_play(self) -> Player:
        return Player()


class Environment(object):
    """The environment MuZero is interacting with."""

    def step(self, action):
        pass


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()  # Game specific environment.
        self.states = []
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.actions = list(range(self.action_space_size))
        # self.env = gym.make('LunarLander-v2')
        self.env = gym.make('CartPole-v1')
        self.states.append(self.env.reset())
        self.done = False

    def terminal(self) -> bool:
        # Game specific termination rules.
        if self.done:
            self.env.close()

        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def apply(self, action: Action):
        observation, reward, done, info = self.env.step(action)
        self.done = done
        self.states.append(observation)
        self.rewards.append(reward)
        self.history.append(action)


    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        #action_space = (Action(index) for index in range(self.action_space_size))
        action_space = list(range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        return np.atleast_2d(self.states[state_index])

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0.
                #last_reward = None

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        batch = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

        return batch


    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(len(game.history))


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(object):
    def __init__(self):
        self.representation: tf.keras.Model
        self.dynamics: tf.keras.Model
        self.prediction: tf.keras.Model

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0

    def get_trainable_weights(self):
        return []


def create_representation_model(config: MuZeroConfig):
    input_shape = (config.env.observation_space.shape[0], )
    inputs = Input(shape=input_shape, name='r_inputs')
    common = Dense(24, activation=tf.nn.relu, name='r_common')(inputs)
    hidden_state = Dense(config.env.observation_space.shape[0], activation=tf.nn.relu, name='r_hidden_state')(common)
    return Model(inputs=inputs, outputs=hidden_state)


def create_dynamics_model(config: MuZeroConfig):
    input_shape = (config.env.observation_space.shape[0] + config.env.action_space.n, )
    inputs = Input(shape=input_shape, name='d_inputs')
    common = Dense(24, activation=tf.nn.relu, name='d_common')(inputs)
    hidden_state = Dense(config.env.observation_space.shape[0], activation=tf.nn.relu, name='d_hidden_state')(common)
    reward = Dense(1, name='d_reward')(common)
    return Model(inputs=inputs, outputs=[hidden_state, reward])


def create_prediction_model(config: MuZeroConfig):
    input_shape = (config.env.observation_space.shape[0], )
    inputs = Input(shape=input_shape, name='p_inputs')
    common = Dense(24, activation=tf.nn.relu, name='p_common')(inputs)
    policy = Dense(config.env.action_space.n, name='p_policy')(common)
    value = Dense(1, name='p_value')(common)
    return Model(inputs=inputs, outputs=[policy, value])


class FCNetwork(Network):
    def __init__(self, config: MuZeroConfig):
        super(FCNetwork).__init__()
        self.config = config
        self.representation = create_representation_model(self.config)
        self.dynamics = create_dynamics_model(self.config)
        self.prediction = create_prediction_model(self.config)
        self.dynamics_checkpoint = tf.train.Checkpoint(model=self.dynamics)
        self.prediction_checkpoint = tf.train.Checkpoint(model=self.prediction)
        self.representation_checkpoint = tf.train.Checkpoint(model=self.representation)

        self.dynamics_checkpoint_path = './data/muzero/saves/Dynamics'
        Path.mkdir(Path(self.dynamics_checkpoint_path), parents=True, exist_ok=True)
        self.manager_dynamics = tf.train.CheckpointManager(self.dynamics_checkpoint,
                                                           directory=self.dynamics_checkpoint_path,
                                                           max_to_keep=5)

        self.prediction_checkpoint_path = './data/muzero/saves/Prediction'
        Path.mkdir(Path(self.prediction_checkpoint_path), parents=True, exist_ok=True)
        self.manager_prediction = tf.train.CheckpointManager(self.prediction_checkpoint,
                                                             directory=self.prediction_checkpoint_path,
                                                             max_to_keep=5)

        self.representation_checkpoint_path = './data/muzero/saves/Representation'
        Path.mkdir(Path(self.representation_checkpoint_path), parents=True, exist_ok=True)
        self.manager_representation = tf.train.CheckpointManager(self.representation_checkpoint,
                                                                 directory=self.representation_checkpoint_path,
                                                                 max_to_keep=5)



        self.load_weights()
        self.min_max_stats = MinMaxStats(config.known_bounds)


    def one_hot_tensor_encoder(self, hidden_state: tf.Tensor, action: Action):
        encoded_action = tf.expand_dims(tf.eye(self.config.env.action_space.n)[action], axis=0)
        encoded_state_action = tf.concat([hidden_state, encoded_action], axis=1)
        return encoded_state_action

    def initial_inference(self, observation) -> NetworkOutput:
        hidden_state = self.representation(observation)
        hidden_state = scale(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        # value = support_to_scalar(value, 10)

        output = NetworkOutput(
            value=value,
            reward=0.,
            policy_logits={i: v for i, v in enumerate(tf.squeeze(policy_logits))},
            hidden_state=hidden_state
        )
        return output

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        encoded_hidden_state = self.one_hot_tensor_encoder(hidden_state, action)
        hidden_state, reward = self.dynamics(encoded_hidden_state)
        hidden_state = scale(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        # value = support_to_scalar(value, 10)

        output = NetworkOutput(
            value=value,
            reward=reward,
            policy_logits={i: v for i, v in enumerate(tf.squeeze(policy_logits))},
            hidden_state=hidden_state
        )
        return output

    def get_trainable_weights(self):
        return self.representation.trainable_weights + self.dynamics.trainable_weights + self.prediction.trainable_weights

    def load_weights(self):
        self.dynamics_checkpoint.restore(self.manager_dynamics.latest_checkpoint)
        self.prediction_checkpoint.restore(self.manager_prediction.latest_checkpoint)
        self.representation_checkpoint.restore(self.manager_representation.latest_checkpoint)

    def save_checkpoint(self):
        self.manager_dynamics.save()
        self.manager_prediction.save()
        self.manager_representation.save()


def latest_file(path: Path, pattern: str = "*/[0-9]*"):
    if path.exists():
        files = path.rglob(pattern)
        if not files:
            return []
        return max(files, key=lambda x: x.stat().st_ctime)
    return []

class SharedStorage(object):

    def __init__(self, config: MuZeroConfig):
        self._networks = {}
        self.config = config

    def latest_network(self) -> FCNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return FCNetwork(self.config)

    def save_network(self, step: int, network: FCNetwork):
        self._networks[step] = network

    def save_network_to_file(self, network: FCNetwork):
        network.save_checkpoint()


##### End Helpers ########
##########################


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    thread_games = Thread(target=run_selfplay, args=(config, replay_buffer, storage))
    thread_games.start()

    while len(replay_buffer.buffer) < 10:
        pass

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, replay_buffer: ReplayBuffer, storage: SharedStorage):

    cpus = cpu_count() - 1
    temperatures = [1.0, 0.5, 0.1]
    step = 0
    with Pool(cpus) as pool:
        while True:
            results = [
                pool.apply_async(
                    func=play_game,
                    args=(config, np.random.choice(temperatures))
                ) for _ in range(cpus)]
            for p in results:
                game = p.get()
                with summary_writer.as_default():
                    tf.summary.scalar('reward', np.sum(game.rewards), step=step)
                replay_buffer.save_game(game)
                step += 1

            storage.save_network_to_file(storage.latest_network())

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, temperature: float) -> Game:
    game = config.new_game()
    network = FCNetwork(config)

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
        action = select_action(config, len(game.history), root, network, temperature)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network, min_max_stats: MinMaxStats):
    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                      config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network, temperature: int = 0):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    return softmax_sample(visit_counts, temperature)


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward +
                                              config.discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########
def preprocess_batch(batch, config: MuZeroConfig):
    # Preprocessing batch data
    observations, actions, targets = zip(*batch)

    new_targets = []

    for target in targets:
        target_value, target_reward, target_policy = zip(*target)

        # value
        target_value = [tf.reshape(tf.Variable(x, dtype=tf.float32), shape=(1, 1)) for x in target_value]

        # reward
        target_reward = [tf.reshape(tf.Variable(x, dtype=tf.float32), shape=(1, 1)) for x in target_reward]

        # policy
        obs_size = config.env.action_space.n
        target_policy = [
            tf.reshape(tf.Variable([0.] * obs_size, dtype=tf.float32), shape=(1, obs_size)) if not v else
            tf.reshape(tf.Variable(v, dtype=tf.float32), shape=(1, obs_size)) for v in target_policy
        ]

        new_targets.append(tuple(zip(target_value, target_reward, target_policy)))

    return list(zip(observations, actions, new_targets))


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = FCNetwork(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_init)

    for i in range(config.training_steps):

        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        batch = preprocess_batch(batch, config)

        def compute_loss():
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

                    policy_logits = tf.stack([list(network_output.policy_logits.values())])

                    l = tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy)

                    l += scalar_loss(network_output.value, target_value)

                    if k > 0:
                        l += scalar_loss(network_output.reward, target_reward)

                    loss += scale_gradient(l, gradient_scale)
            loss /= config.batch_size

            for weights in network.get_weights():
                loss += config.weight_decay * tf.nn.l2_loss(weights)

            loss_total(loss)
            return loss

        optimizer.minimize(compute_loss, network.get_trainable_weights())

        # grads = tape.gradient(loss, trainable_weights)
        # optimizer.apply_gradients(zip(grads, trainable_weights))

        with summary_writer.as_default():
            tf.summary.scalar('loss total', loss_total.result(), step=i)

        loss_total.reset_state()


    # storage.save_network(config.training_steps, network)


def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights2(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch,
                   weight_decay: float):
    loss = tf.Variable(0.)
    trainable_weights = network.get_trainable_weights()

    with tf.GradientTape() as tape:
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            value, reward, policy_logits, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_logits)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))
                hidden_state = scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):

                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                # Value Loss
                value_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                    y_true=scalar_to_support(target_value, 10),
                    y_pred=scalar_to_support(value, 10)
                )

                # Reward Loss
                reward_loss = scalar_loss(reward, target_reward)

                # Policy Loss
                policy_logits = tf.stack([list(policy_logits.values())])
                policy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                    y_true=target_policy,
                    y_pred=policy_logits
                )

                loss_value(value_loss)
                loss_reward(reward_loss)
                loss_policy(policy_loss)

                l = tf.add(tf.add(value_loss, reward_loss), policy_loss)

                loss = tf.add(loss, scale_gradient(l, gradient_scale))
                # loss += scale_gradient(l, gradient_scale)

            for weights in trainable_weights:
                loss = tf.add(loss, weight_decay * tf.nn.l2_loss(weights))

        ic(loss)
        loss_total(loss)
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))



def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return tf.reduce_sum(tf.square(tf.cast(target, tf.float32) - tf.cast(prediction, tf.float32)))


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float = 0.3):
    # helper function to sample an index from a probability array
    d = np.asarray([x for x, y in distribution])
    counts = d ** (1 / temperature)
    p = counts / sum(counts)
    action = np.random.choice(len(counts), p=p)
    return distribution[action][1]


if __name__ == "__main__":
    np.random.seed(8273)

    loss_value = tf.keras.metrics.Mean('loss_value', dtype=tf.float32)
    loss_reward = tf.keras.metrics.Mean('loss_reward', dtype=tf.float32)
    loss_policy = tf.keras.metrics.Mean('loss_policy', dtype=tf.float32)

    train_score = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    loss_total = tf.keras.metrics.Mean('loss_total', dtype=tf.float32)

    data_path = Path("./data/muzero")
    data_path.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")

    muzero(make_config())
