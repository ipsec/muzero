import ray

from buffers import ReplayBuffer
from games.game import Game
from mcts import expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from config import MuZeroConfig
from stats import Summary
from storage import SharedStorage
from utils import MinMaxStats, Node


@ray.remote
class Actor:
    def __init__(self, config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, summary: Summary):
        self.config = config
        self.network = Network(self.config)
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.summary = summary
        self.games = 0

    def update_value(self, weights):
        self.network.set_weights(weights)

    def start(self):
        while self.network.training_steps() < 100000:
            game = self.play_game()
            self.games += 1
            self.summary.games.remote()
            self.summary.reward.remote(game.get_reward_total())
            self.summary.publish_games.remote(self.games)
            self.replay_buffer.save_game.remote(game)
            if ray.get(self.storage.started.remote()):
                weights = ray.get(self.storage.get_network_weights.remote())
                self.network.set_weights(weights)

    def play_game(self) -> Game:
        game = Game(self.config.discount)
        min_max_stats = MinMaxStats(self.config.known_bounds)

        while not game.terminal() and len(game.history) < self.config.max_moves:

            # At the root of the search tree we use the representation function to
            # obtain a hidden state given the current observation.
            root = Node(0)
            current_observation = game.get_observation_from_index(-1)
            network_output = self.network.initial_inference(current_observation)
            expand_node(root, game.to_play(), game.legal_actions(), network_output)
            backpropagate([root], network_output.value, game.to_play(), self.config.discount, min_max_stats)
            add_exploration_noise(self.config, root)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            run_mcts(self.config, root, game.action_history(), self.network, min_max_stats)
            action = select_action(root, self.network, self.config)
            game.apply(action)
            game.store_search_statistics(root)

        return game
