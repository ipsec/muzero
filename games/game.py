from typing import List

import gym
import numpy as np
from gym import Env

from config import MuZeroConfig
from utils import Node


class Player(object):
    def __init__(self):
        self.player = 'Player 0'

    def __eq__(self, other):
        return True


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


class Environment:

    def __init__(self):
        # self.env = gym.make('LunarLander-v2')
        self.env = gym.make('CartPole-v1')
        self.action_space_size = self.env.action_space.n

    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        return self.env.reset()


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, discount: float):
        self.env = Environment()
        self.observations = [self.reset()]
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = self.env.action_space_size
        self.discount = discount
        self.done = False
        self.actions = list(map(lambda i: Action(i), range(self.action_space_size)))

    def terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def apply(self, action: Action):
        observation, reward, done, info = self.env.step(action.index)
        self.done = done

        if self.done:
            self.close()

        self.observations.append(observation)
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

    def reset(self):
        observation = self.env.reset()
        return observation

    def render(self):
        self.env.render()

    def get_observation_from_index(self, state_index: int):
        return self.observations[state_index]

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

    def close(self):
        self.env.close()
