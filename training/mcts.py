import math

import tensorflow as tf
from tensorflow.keras.losses import MSE

import numpy as np
from typing import List, Any



##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
from config import MuZeroConfig
from core.core import SharedStorage, ReplayBuffer


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    returns = []
    count = 0
    for _ in range(2):
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

                print(current_policy_logits)
                print(target_policy)

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
