from typing import List

import gym
import numpy as np

from config import MuZeroConfig
from utils import Node


class Player(object):
    pass


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


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

    def __init__(self, action_space_size: int, discount: float):
        self.env = gym.make('LunarLander-v2')
        self.states = [self.env.reset()]
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.done = False

    def terminal(self) -> bool:
        # Game specific termination rules.
        return self.done

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
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
        # Game specific feature planes.
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
                last_reward = None

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
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(len(game.history))


def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        state_space_size=8,
        action_space_size=4,
        max_moves=1000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=15,
        batch_size=2,
        td_steps=10,
        num_actors=1,
        num_episodes=4,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)
