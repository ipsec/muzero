from typing import Optional

from gym import Env

from utils import KnownBounds


class MuZeroConfig(object):

    def __init__(self,
                 env: Env,
                 state_space_size: int,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 lr_decay_rate: float,
                 training_steps: int = int(1000e3),
                 checkpoint_interval: int = 10,
                 save_interval: int = 10,
                 known_bounds: Optional[KnownBounds] = None):
        # Environment
        self.env = env

        # Self-Play
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.num_actors = num_actors

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

        # Training
        self.training_steps = training_steps
        self.checkpoint_interval = 10
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.checkpoint_interval = checkpoint_interval
        self.support_size = 15
        self.save_interval = save_interval

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        # Exponential decay factor temperature
        self.temperature_decay_factor = 1e-4
        self.temperature_min = 0.1
