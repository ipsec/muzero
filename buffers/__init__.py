import ray
import numpy as np
from collections import deque

from config import MuZeroConfig
from games.game import Game


@ray.remote
class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = deque(maxlen=self.window_size)

    def save_game(self, game):
        self.buffer.append(game)

    def sample_batch(self):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.get_observation_from_index(i), g.history[i:i + self.config.num_unroll_steps],
                 g.make_target(i, self.config.num_unroll_steps, self.config.td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)
        # return choice(self.buffer)

    def sample_position(self, game) -> int:
        return np.random.choice(len(game.history))
        # return randrange(len(game.history))

    def size(self):
        return len(self.buffer)
