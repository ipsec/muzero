import ray
import tensorflow as tf

from tensorflow.keras.metrics import Sum, Mean
from buffers import ReplayBuffer
from config import MuZeroConfig
from games.game import Game
from mcts import expand_node, backpropagate, add_exploration_noise, run_mcts, select_action
from models.network import Network
from stats import create_summary
from storage import SharedStorage
from utils import MinMaxStats, Node


@ray.remote
class Actor:
    def __init__(self,
                 config: MuZeroConfig,
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer,
                 temperature: float = 1.0):
        self.config = config
        self.network = Network(self.config)
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.temperature = temperature
        self.name = f"games-{temperature}"
        self.summary = create_summary(name=self.name)
        self.games_played = 0
        self.metrics_games = Sum(self.name, dtype=tf.int32)
        self.metrics_temperature = Sum(self.name, dtype=tf.float32)
        self.metrics_rewards = Mean(self.name, dtype=tf.float32)
        self.started = False

    def update_metrics(self):
        with self.summary.as_default():
            tf.summary.scalar(f'games-played', self.metrics_games.result(), self.games_played)
            tf.summary.scalar(f'games-temperature', self.metrics_temperature.result(), self.games_played)
            tf.summary.scalar(f'games-rewards', self.metrics_rewards.result(), self.games_played)

        self.metrics_temperature.reset_states()
        self.metrics_rewards.reset_states()

    def start(self):
        while self.games_played < self.config.training_steps:
            game = self.play_game()
            self.games_played += 1
            self.metrics_games(1)
            self.metrics_rewards(sum(game.rewards))
            self.update_metrics()

            self.replay_buffer.save_game.remote(game)

            if not self.started:
                self.started = ray.get(self.storage.started.remote())
                continue

            weights = ray.get(self.storage.get_network_weights.remote())
            self.network.set_weights(weights)

        print(f"Actor: {self.name } finished.")

    def play_game(self) -> Game:
        game = Game(self.config.discount)
        min_max_stats = MinMaxStats(self.config.known_bounds)

        # Use Exponential Decay to reduce temperature over time
        temperature = max(
            self.temperature * (1 - self.config.temperature_decay_factor) ** self.network.training_steps(),
            self.config.temperature_min
        )
        self.metrics_temperature(temperature)

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
            action = select_action(root, temperature)
            game.apply(action)
            game.store_search_statistics(root)

        return game
