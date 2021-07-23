# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

import collections
import random
import typing
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
import math
import numpy as np
import tensorflow as tf
from icecream import ic
from tensorflow.keras.losses import MSE
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

##########################
####### Helpers ##########


MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def scale(t: tf.Tensor):
    # return (t - np.min(t)) / (np.max(t) - np.min(t))
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


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
        self.window_size = 1000
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

    def new_game(self, max_steps: int = 500):
        return Game(self.action_space_size, self.discount, max_steps)


def make_config() -> MuZeroConfig:
    return MuZeroConfig(
        max_moves=500,
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=25,
        batch_size=32,
        td_steps=1000,
        num_actors=1,
        lr_init=0.01,
        lr_decay_steps=350e3,
        # known_bounds=KnownBounds(0, 1),
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
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()


class Environment(object):
    """The environment MuZero is interacting with."""

    def step(self, action):
        pass


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float, max_steps: int = 500):
        self.max_steps = max_steps
        self.environment = Environment()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.actions = list(map(lambda i: Action(i), range(self.action_space_size)))
        # self.env = gym.make('LunarLander-v2')
        self.env = gym.make('CartPole-v1')
        self.states = [self.env.reset()]
        self.done = False

    def terminal(self) -> bool:
        # Game specific termination rules.
        if len(self.history) >= self.max_steps:
            self.done = True

        if self.done:
            self.env.close()

        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def apply(self, action: Action):
        observation, reward, done, info = self.env.step(action.index)
        self.done = done
        self.states.append(observation)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        return self.states[state_index]

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
        # Generate some sample of data to train on
        games = self.sample_games()
        game_pos = [(g, self.sample_position(g)) for g in games]
        game_data = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                      g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                     for (g, i) in game_pos]

        # Pre-process the batch
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=None))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT
        mask_time_batch = []
        dynamic_mask_time_batch = []
        last_mask = [True] * len(image_batch)
        for i, actions_batch in enumerate(actions_time_batch):
            mask = list(map(lambda a: bool(a), actions_batch))
            dynamic_mask = [now for last, now in zip(last_mask, mask) if last]
            mask_time_batch.append(mask)
            dynamic_mask_time_batch.append(dynamic_mask)
            last_mask = mask
            actions_time_batch[i] = [action.index for action in actions_batch if action]

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch
        return batch

    def sample_games(self) -> List[Game]:
        # Sample game from buffer either uniformly or according to some priority.
        return random.choices(self.buffer, k=self.batch_size)

    def sample_position(self, game: Game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return random.randint(0, len(game.history))

"""
    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)
        # return choice(self.buffer)

    def sample_position(self, game) -> int:
        return np.random.choice(len(game.history))
        # return randrange(len(game.history))
"""


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


def create_representation_model(config: MuZeroConfig, weight_decay: float = 1e-4):
    input_size = config.env.observation_space.shape[0]
    model = tf.keras.models.Sequential()
    model.add(Dense(input_size, activation=tf.nn.relu, name="representation_input"))
    model.add(Dense(64, activation=tf.nn.relu, name='representation_output'))
    return model


def create_dynamics_model(config: MuZeroConfig, weight_decay: float = 1e-4):
    # input_size = config.env.observation_space.shape[0] + config.env.action_space.n
    input_size = 512 + config.env.action_space.n
    output_size = config.env.observation_space.shape[0]
    model = tf.keras.models.Sequential()
    model.add(Dense(input_size, activation=tf.nn.relu, name="dynamics_input"))
    model.add(Dense(64, name='dynamics_output', activation=tf.nn.relu))
    return model


def create_value_model(config: MuZeroConfig, weight_decay: float = 1e-4):
    input_size = config.env.observation_space.shape[0]
    model = tf.keras.models.Sequential()
    model.add(Dense(input_size, activation=tf.nn.relu, name="value_input"))
    model.add(Dense(101, activation=tf.nn.softmax, name='value_output'))
    return model


def create_policy_model(config: MuZeroConfig, weight_decay: float = 1e-4):
    input_size = config.env.observation_space.shape[0]
    output_size = config.env.action_space.n
    model = tf.keras.models.Sequential()
    model.add(Dense(input_size, activation=tf.nn.relu, name="policy_input"))
    model.add(Dense(output_size, name='policy_output'))
    return model


def create_reward_model(config: MuZeroConfig, weight_decay: float = 1e-4):
    input_size = config.env.observation_space.shape[0] + config.env.action_space.n
    model = tf.keras.models.Sequential()
    model.add(Dense(input_size, activation=tf.nn.relu, name="reward_input"))
    model.add(Dense(1, name='reward_output'))
    return model


class InitialModel(Model):
    """Model that combine the representation and prediction (value+policy) network."""

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, image, training: bool = False, **kwargs):
        hidden_representation = self.representation_network(image, training=training)
        hidden_representation = scale(hidden_representation)
        value = self.value_network(hidden_representation, training=training)
        policy_logits = self.policy_network(hidden_representation, training=training)
        return hidden_representation, value, policy_logits


class RecurrentModel(Model):
    """Model that combine the dynamic, reward and prediction (value+policy) network."""

    def __init__(self, dynamic_network: Model, reward_network: Model, value_network: Model, policy_network: Model):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden, training: bool = False, **kwargs):
        hidden_state = self.dynamic_network(conditioned_hidden, training=training)
        reward = self.reward_network(conditioned_hidden, training=training)
        value = self.value_network(hidden_state, training=training)
        policy_logits = self.policy_network(hidden_state, training=training)
        hidden_state = scale(hidden_state)
        return hidden_state, reward, value, policy_logits


class FCNetwork(Network):
    # policy -> uniform, value -> 0, reward -> 0

    def __init__(self, config: MuZeroConfig):
        super(FCNetwork).__init__()
        self.config = config
        self.representation = create_representation_model(self.config)
        self.dynamics = create_dynamics_model(self.config)
        # self.prediction = create_prediction_model(self.config)
        self.value = create_value_model(self.config)
        self.policy = create_policy_model(self.config)
        self.reward = create_reward_model(self.config)
        self.initial_model = InitialModel(self.representation, self.value, self.policy)
        self.recurrent_model = RecurrentModel(self.dynamics, self.reward, self.value, self.policy)

        # self.dynamics_checkpoint = tf.train.Checkpoint(model=self.dynamics)
        # self.prediction_checkpoint = tf.train.Checkpoint(model=self.prediction)
        # self.representation_checkpoint = tf.train.Checkpoint(model=self.representation)

        # self.dynamics_checkpoint_path = './data/muzero/saves/Dynamics'
        # Path.mkdir(Path(self.dynamics_checkpoint_path), parents=True, exist_ok=True)
        # self.manager_dynamics = tf.train.CheckpointManager(self.dynamics_checkpoint,
        # directory=self.dynamics_checkpoint_path,
        # max_to_keep=5)

        # self.prediction_checkpoint_path = './data/muzero/saves/Prediction'
        # Path.mkdir(Path(self.prediction_checkpoint_path), parents=True, exist_ok=True)
        # self.manager_prediction = tf.train.CheckpointManager(self.prediction_checkpoint,
        # directory=self.prediction_checkpoint_path,
        # max_to_keep=5)

        # self.representation_checkpoint_path = './data/muzero/saves/Representation'
        # Path.mkdir(Path(self.representation_checkpoint_path), parents=True, exist_ok=True)
        # self.manager_representation = tf.train.CheckpointManager(self.representation_checkpoint,
        # directory=self.representation_checkpoint_path,
        # max_to_keep=5)

        # self.load_weights()
        self.min_max_stats = MinMaxStats(config.known_bounds)

    def initial_inference(self, image: np.array, training: bool = False) -> NetworkOutput:
        """representation + prediction function"""

        hidden_representation, value, policy_logits = self.initial_model(np.expand_dims(image, 0), training)

        output = NetworkOutput(value=tf_support_to_scalar(value, 50),
                               reward=0.,
                               policy_logits={Action(i): float(logit) for i, logit in enumerate(policy_logits[0])},
                               hidden_state=hidden_representation[0].numpy())
        return output

    def recurrent_inference(self, hidden_state: np.array, action: Action, training: bool = False) -> NetworkOutput:
        """dynamics + prediction function"""

        conditioned_hidden = self.get_conditioned_hidden(hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model(conditioned_hidden, training)
        output = NetworkOutput(value=tf_support_to_scalar(value, 50),
                               reward=float(reward),
                               policy_logits={Action(i): float(logit) for i, logit in enumerate(policy_logits[0])},
                               hidden_state=hidden_representation[0].numpy())
        return output

    def get_conditioned_hidden(self, hidden: np.ndarray, action: Action):
        conditioned_hidden = tf.concat((hidden, tf.eye(self.config.env.action_space.n)[action.index]), axis=0)
        conditioned_hidden = tf.expand_dims(conditioned_hidden, axis=0)
        return conditioned_hidden

    def get_trainable_weights(self):
        return self.representation.trainable_weights + self.dynamics.trainable_weights + self.value.trainable_weights + self.reward.trainable_weights + self.policy.trainable_weights

    def get_weights(self):
        return self.get_trainable_weights()

    def get_trainable_variables(self):
        return self.representation.trainable_variables + self.dynamics.trainable_variables + self.value.trainable_variables + self.reward.trainable_variables + self.policy.trainable_variables

    def load_weights(self):
        dynamics_latest_checkpoint = self.manager_dynamics.restore_or_initialize()
        prediction_latest_checkpoint = self.manager_prediction.restore_or_initialize()
        representation_latest_checkpoint = self.manager_representation.restore_or_initialize()

        self.dynamics_checkpoint.restore(dynamics_latest_checkpoint)
        self.prediction_checkpoint.restore(prediction_latest_checkpoint)
        self.representation_checkpoint.restore(representation_latest_checkpoint)

    def save_checkpoint(self):
        self.manager_dynamics.save()
        self.manager_prediction.save()
        self.manager_representation.save()


@tf.function
def tf_scalar_to_support(x: tf.Tensor, support_size: int, eps: float = 0.001) -> tf.Tensor:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + eps * x

    transformed = tf.clip_by_value(x, -support_size, support_size - 1e-6)
    floored = tf.floor(transformed)
    prob = transformed - floored  # Proportion between adjacent integers

    idx_0 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size), dtype=tf.int32), -1)
    idx_1 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size + 1), dtype=tf.int32), -1)
    idx_0 = tf.stack([tf.range(x.shape[1]), idx_0])
    idx_1 = tf.stack([tf.range(x.shape[1]), idx_1])
    indexes = tf.squeeze(tf.stack([idx_0, idx_1]))

    updates = tf.squeeze(tf.concat([1 - prob, prob], axis=0))
    return tf.scatter_nd(indexes, updates, (1, 2 * support_size + 1))


@tf.function
def tf_support_to_scalar(x: tf.Tensor, support_size: int, eps: float = 0.001) -> tf.Tensor:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # x = tf.nn.softmax(x)

    bins = tf.range(-support_size, support_size + 1, dtype=tf.float32)
    value = tf.tensordot(tf.squeeze(x), tf.squeeze(bins), 1)

    value = tf.math.sign(value) * (
                ((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(value) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
    return value

def support_to_scalar(x: np.ndarray, support_size: int, var_eps: float = 0.001) -> np.ndarray:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    x = tf.nn.softmax(x)

    bins = np.arange(-support_size, support_size + 1)
    y = np.dot(x, bins)

    value = np.sign(y) * (((np.sqrt(1 + 4 * var_eps * (np.abs(y) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)

    return value.item()


def scalar_to_support(x: np.ndarray, support_size: int, var_eps: float = 0.001) -> np.ndarray:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Clip float to fit within the support_size. Values exceeding this will be assigned to the closest bin.

    transformed = np.clip(np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x,
                          a_min=-support_size, a_max=support_size - 1e-8)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored  # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins


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
    network = storage.latest_network()
    storage.save_network(0, network)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_init)
    step = 10

    for i in range(100000):
        network = storage.latest_network()
        run_selfplay(config, replay_buffer, storage, i)
        if i % step == 0:
            train_network(config, network, optimizer, replay_buffer, int(i / step))
            # storage.save_network(i, network)
            # storage.save_network_to_file(storage.latest_network())

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, replay_buffer: ReplayBuffer, storage: SharedStorage, step: int = 0):
    game = play_game(config, storage, 1.0)
    with summary_writer.as_default():
        tf.summary.scalar('reward', np.sum(game.rewards), step=step)
    summary_writer.flush()
    replay_buffer.save_game(game)


"""
    cpus = cpu_count() - 1
    temperatures = list(np.linspace(1.0, 0.1, num=cpus))
    step = 0
    with Pool(cpus) as pool:
        while True:
            results = [pool.apply_async(func=play_game, args=(config, temperatures[i]))
                       for i, _ in enumerate(range(cpus))]
            for p in results:
                game = p.get()
                with summary_writer.as_default():
                    tf.summary.scalar('reward', np.sum(game.rewards), step=step)
                replay_buffer.save_game(game)
                step += 1

            storage.save_network_to_file(storage.latest_network())
"""


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, storage: SharedStorage, temperature: float) -> Game:
    game = config.new_game(max_steps=config.max_moves)
    network = storage.latest_network()

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


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network, temperature: float = 0.35):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
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

def train_network(config: MuZeroConfig, network: FCNetwork, optimizer: tf.keras.optimizers.Optimizer,
                  replay_buffer: ReplayBuffer, step: int):
    # for i in range(config.training_steps):
    # if i % config.checkpoint_interval == 0:
    #    storage.save_network(i, network)

    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights2(optimizer, network, batch, config, step)


def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights2(optimizer: tf.keras.optimizers, network: FCNetwork, batch, config: MuZeroConfig, step: int):
    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor

    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch = network.initial_model(np.array(image_batch))

        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = tf.boolean_mask(policy_batch, mask_policy)

        # Compute the loss of the first pass
        loss += tf.math.reduce_mean(loss_value(target_value_batch, value_batch, 101))
        loss += tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                    mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # Only execute BPTT for elements with an action
            representation_batch = tf.boolean_mask(representation_batch, dynamic_mask)
            target_value_batch = tf.boolean_mask(target_value_batch, mask)
            target_reward_batch = tf.boolean_mask(target_reward_batch, mask)
            # Creating conditioned_representation: concatenate representations with actions batch
            actions_batch = tf.one_hot(actions_batch, config.env.action_space.n)

            # Recurrent step from conditioned representation: recurrent + prediction networks
            conditioned_representation_batch = tf.concat((representation_batch, actions_batch), axis=1)
            representation_batch, reward_batch, value_batch, policy_batch = network.recurrent_model(
                conditioned_representation_batch)

            # Only execute BPTT for elements with a policy target
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = tf.convert_to_tensor([policy for policy in target_policy_batch if policy])
            policy_batch = tf.boolean_mask(policy_batch, mask_policy)

            # Compute the partial loss
            l = (tf.math.reduce_mean(loss_value(target_value_batch, value_batch, 101)) +
                 MSE(target_reward_batch, tf.squeeze(reward_batch)) +
                 tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)

        loss_total(loss)
        return loss

    optimizer.minimize(loss=loss, var_list=network.get_trainable_variables())

    with summary_writer.as_default():
        tf.summary.scalar('loss total', loss_total.result(), step=step)

    loss_total.reset_state()


#@tf.function
def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: FCNetwork, batch, weight_decay: float, step: int):
    loss = tf.constant(0., dtype=tf.float32)
    trainable_variables = network.get_trainable_variables()

    with tf.GradientTape() as tape:
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            observation = tf.expand_dims(image, 0)
            hidden_state, value, policy_logits = network.initial_model(observation, training=True)
            predictions = [(1.0, value, 0., policy_logits)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                conditioned_hidden = network.get_conditioned_hidden(hidden_state[0], action)
                hidden_state, reward, value, policy_logits = network.recurrent_model(conditioned_hidden, training=True)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))
                hidden_state = scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):

                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                if not target_policy:
                    policy_loss = tf.stop_gradient(tf.constant(0., dtype=tf.float32))
                else:
                    policy_logits = policy_logits[0]
                    target_policy = tf.convert_to_tensor(target_policy)
                    policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_policy, logits=policy_logits)

                if not target_reward:
                    reward_loss = tf.stop_gradient(tf.constant(0., dtype=tf.float32))
                else:
                    reward_loss = scalar_loss(prediction=tf.convert_to_tensor([reward]),
                                              target=tf.convert_to_tensor([target_reward]))

                # Value Loss
                value_loss = tf.squeeze(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf_scalar_to_support(tf.reshape(
                        tf.convert_to_tensor(target_value, dtype=tf.float32), shape=(1, 1)
                    ), 50),
                    logits=value,
                ))

                l = tf.constant(0., dtype=tf.float32)
                l = tf.add(l, policy_loss)
                l = tf.add(l, value_loss)
                l = tf.add(l, reward_loss)

                loss = tf.add(loss, scale_gradient(l, gradient_scale))

            for weights in network.get_weights():
                loss = tf.add(loss, weight_decay * tf.nn.l2_loss(weights))

        loss = tf.divide(loss, len(batch))
        grads = tape.gradient(loss, trainable_variables)
        ic(loss)

    optimizer.apply_gradients(zip(grads, trainable_variables))
    loss_total(loss)

    with summary_writer.as_default():
        tf.summary.scalar('loss total', loss_total.result(), step=step)

    loss_total.reset_state()

def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, value_support_size))
    sqrt_value = np.sqrt(target_value_batch)
    floor_value = np.floor(sqrt_value).astype(int)
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)

def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    prediction = tf.reshape(prediction, shape=(1,))
    target = tf.reshape(target, shape=(1,))
    return tf.losses.mean_squared_error(target, prediction)


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

    #loss_value = tf.keras.metrics.Mean('loss_value', dtype=tf.float32)
    #loss_reward = tf.keras.metrics.Mean('loss_reward', dtype=tf.float32)
    #loss_policy = tf.keras.metrics.Mean('loss_policy', dtype=tf.float32)

    train_score = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    loss_total = tf.keras.metrics.Mean('loss_total', dtype=tf.float32)

    data_path = Path("./data/muzero")
    data_path.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")

    muzero(make_config())
