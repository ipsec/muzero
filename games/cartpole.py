import gym

from core.core import Action, Player, Node, ActionHistory


class Game(object):
    def __init__(self, seed=None, discount: float = 0.95):
        self.env = gym.make("CartPole-v1")
        if seed is not None:
            self.env.seed(seed)

        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = self.env.action_space.n
        self.discount = discount
        self.done = False
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]

    def apply(self, action: Action):
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
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

    def step(self, action):
        """Execute one step of the game conditioned by the given action."""

        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self) -> bool:
        return self.done

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def legal_actions(self) -> list:
        return self.actions

    def store_search_statistics(self, root: Node) -> None:
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)
