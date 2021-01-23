import math

import gym
import numpy as np
from random import choice, randint

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
from typing import Optional, NamedTuple, List, Dict, Any

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


class MuZeroConfig(object):

    def __init__(self,
                 action_space_size: int,
                 observation_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None):
        ### Self-Play
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
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
        self.training_steps = 500
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(seed=42, discount=self.discount)


def make_cartpole_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        return 1.0

    return MuZeroConfig(
        observation_space_size=4,
        num_actors=1,
        lr_init=0.001,
        lr_decay_steps=5,
        action_space_size=2,
        max_moves=500,
        discount=0.99,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        batch_size=512,
        td_steps=10,
        visit_softmax_temperature_fn=visit_softmax_temperature)


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class RepresentationOutput(NamedTuple):
    state_zero: List[Tensor]


class DynamicsOutput(NamedTuple):
    state_k: List[Tensor]
    reward_k: Tensor


class PredictionOutput(NamedTuple):
    policy_logits: List[Tensor]
    value: Tensor


class Representation(Model):
    def __init__(self, observation_space_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param observation_space_size
        """
        super(Representation, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.s0 = Dense(observation_space_size, activation='relu')

    @tf.function
    def call(self, observation, **kwargs) -> RepresentationOutput:
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        x = self.common(x)
        s_0 = self.s0(x)
        return RepresentationOutput(state_zero=s_0)


class Dynamics(Model):
    def __init__(self, observation_space_size: int):
        """
        r^k, s^k = g_0(s^(k-1), a^k)
        :param observation_space_size: size of observation_space_size
        """
        super(Dynamics, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.s_k = Dense(observation_space_size, activation='relu')
        self.r_k = Dense(1, activation='relu')

    @tf.function
    def call(self, encoded_space, **kwargs) -> DynamicsOutput:
        """
        :param encoded_space: hidden state concatenated with one_hot action
        :return: DynamicsOutput with reward (r^k) and hidden state (s^k)
        """
        x = self.inputs(encoded_space)
        x = self.hidden(x)
        x = self.common(x)
        s_k = self.s_k(x)
        r_k = self.r_k(x)
        return DynamicsOutput(reward_k=r_k, state_k=s_k)


class Prediction(Model):
    def __init__(self, action_space_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_space_size
        """
        super(Prediction, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.policy = Dense(action_space_size, activation='relu')
        self.value = Dense(1, activation='relu')

    @tf.function
    def call(self, hidden_state, **kwargs) -> PredictionOutput:
        """
        :param hidden_state
        :return:PredictionOutput with policy_logits and value
        """
        x = self.inputs(hidden_state)
        x = self.hidden(x)
        x = self.common(x)
        policy = self.policy(x)
        value = self.value(x)
        return PredictionOutput(policy_logits=policy, value=value)


class Network(object):

    def __init__(self, observation_space_size: int, action_space_size: int):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.h = Representation(self.observation_space_size)
        self.g = Dynamics(self.observation_space_size)
        self.f = Prediction(self.action_space_size)
        self.training_steps = 0

    def initial_inference(self, observation) -> NetworkOutput:
        """
        Esse metodo é chamado somente no nó raiz da árvore, por isso a reward é 0.
        :param observation
        :return: NetworkOutput
        """

        h_output = self.h(np.atleast_2d(observation))
        f_output = self.f(h_output.state_zero)

        # reward equal to 0 for consistency
        r_0 = 0.

        policy_logits = NetworkOutput.build_policy_logits(f_output.policy_logits)
        return NetworkOutput(
            value=f_output.value,
            reward=r_0,
            policy_logits=policy_logits,
            hidden_state=h_output.state_zero)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        encoded_space = self.get_encoded_space(hidden_state, action)
        g_output = self.g(encoded_space)
        f_output = self.f(g_output.state_k)

        policy_logits = NetworkOutput.build_policy_logits(f_output.policy_logits)

        return NetworkOutput(
            value=f_output.value,
            reward=g_output.reward_k,
            policy_logits=policy_logits,
            hidden_state=g_output.state_k)

    def get_encoded_space(self, state, action) -> Tensor:
        """
        A entrada para a função g é o resultado do one_hot concatenado ao hidden_state
        conforme 'Observation and action encoding' do documento
        :param state: hidden_state
        :param action:
        :return:
        """
        one_hot = np.atleast_2d(tf.one_hot(action.index, self.action_space_size))
        return tf.concat([state, one_hot], axis=1)

    def get_weights(self) -> list:
        return [self.h.get_weights(), self.g.get_weights(), self.f.get_weights()]

    def get_variables(self) -> list:
        networks = (self.h, self.g, self.f)
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list]


def make_uniform_network(config: MuZeroConfig):
    return Network(config.observation_space_size, config.action_space_size)


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

    def sample_game(self):
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


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    returns = []
    count = 0
    for _ in range(config.training_steps):
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        returns.append(sum(game.rewards))
        count += 1
    print(network.training_steps)
    print(f"Count: {count} - Rewards: {np.sum(returns) / network.training_steps}")


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        initial_observation = network.initial_inference(current_observation)
        expand_node(root, game.to_play(), game.legal_actions(), initial_observation)
        backpropagate([root], initial_observation.value, game.to_play(), config.discount, min_max_stats)
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network, min_max_stats)
        action = select_action(config, len(game.history), root, network)
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
                  network: Network):
    """
    After running simulations inside in MCTS, we select an action based on the root's children visit counts.
    During training we use a softmax sample for exploration.
    During evaluation we select the most visited child.
    """
    visit_counts = [child.visit_count for child in node.children.values()]
    actions = [action for action in node.children.keys()]
    action = None
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps)
    action = softmax_sample(visit_counts, actions, t)

    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child


# %%

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
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


# %%

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# %%

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# %%

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


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = Network(config.observation_space_size, config.action_space_size)
    optimizer = tf.optimizers.Adam()

    for _ in range(config.training_steps):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(network.training_steps, network)


def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: tf.optimizers, network: Network, batch, weight_decay: float):
    """
        with tf.GradientTape() as tape:
        selected_action_values = tf.math.reduce_sum(
            self.model(np.atleast_2d(states)) * tf.one_hot(actions, self.action_size), axis=1)
        loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
    variables = self.model.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    """

    with tf.GradientTape() as tape:
        loss = 1e-10
        for observation, actions, targets in batch:
            # Initial step, from the real observation.
            network_output = network.initial_inference(observation)
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
                current_policy_logits = tf.concat(list(network_output.policy_logits.values()), axis=0)

                # TODO: target_policy receiving empty list
                l = tf.nn.softmax_cross_entropy_with_logits(logits=current_policy_logits, labels=target_policy)
                l += scalar_loss(network_output.value, target_value)
                if k > 0:
                    l += scalar_loss(network_output.reward, target_reward)
                loss += scale_gradient(l, gradient_scale)

        loss /= len(batch)

        for weights in network.get_weights():
            partial_loss = np.sum([tf.nn.l2_loss(v) * weight_decay for v in weights]) / len(weights)
            loss += partial_loss

    variables = network.get_variables()
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    network.training_steps += 1


def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return MSE(prediction, target)


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        run_selfplay(config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


if __name__ == '__main__':
    config = make_cartpole_config()
    muzero(config)
