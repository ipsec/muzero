from collections import deque
from random import randrange, choice
from typing import List

import gym
import numpy as np
from gym import Env

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
    def __init__(self):
        # self.env = gym.make('LunarLander-v2')
        self.env = gym.make('CartPole-v1')
        self.action_space_size = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def close(self):
        self.env.close()


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, discount: float):
        self.env = Environment()
        self.states = [self.env.reset()]
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = self.env.action_space_size
        self.discount = discount
        self.actions = list(map(lambda i: Action(i), range(self.action_space_size)))
        self.done = False

    def terminal(self) -> bool:
        # Game specific termination rules.
        if self.done:
            self.env.close()

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

            if 0 < current_index <= len(self.rewards):
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
        self.buffer = deque(maxlen=self.window_size)
        self.buffer_tmp = deque(maxlen=self.window_size)

    def update_main(self):
        """
        Update self.buffer with recent played games
        :return: None
        """
        self.buffer += self.buffer_tmp
        self.buffer_tmp.clear()

    def save_game(self, game):
        self.buffer_tmp.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)
        #return choice(self.buffer)

    def sample_position(self, game) -> int:
        return np.random.choice(len(game.history))
        #return randrange(len(game.history))


def make_atari_config(env: Env) -> MuZeroConfig:

    return MuZeroConfig(
        env=env,
        state_space_size=int(np.prod(env.observation_space.shape)),
        action_space_size=env.action_space.n,
        max_moves=700,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=15,  # Number of future moves self-simulated
        batch_size=32,
        td_steps=10,  # Number of steps in the future to take into account for calculating the target value
        num_actors=10,
        training_steps=10000,
        checkpoint_interval=100,
        lr_init=0.02,
        lr_decay_steps=1000,
        lr_decay_rate=0.9)
