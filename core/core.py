import numpy as np
from random import choice, randint
from typing import List, Optional, NamedTuple


MAXIMUM_FLOAT_VALUE = float('inf')


class KnownBounds(NamedTuple):
    min: float
    max: float


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
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        return choice(self.buffer)

    def sample_position(self, game) -> int:
        return randint(0, len(game.history))


class SharedStorage(object):

    def __init__(self, config: MuZeroConfig):
        self._networks = {}
        self.config = config

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.config)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def scale(x):
    # reward e value using an invertible transform h(x) = sign(x)(sqrt(|x| + 1) − 1) + εx,
    eps = 0.0001
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x


def softmax_sample(visit_counts, actions, t):
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]


def launch_job(f, *args):
    f(*args)


def make_uniform_network(config: MuZeroConfig):
    return Network(config.observation_space_size, config.action_space_size)
