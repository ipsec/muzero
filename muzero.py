# -*- coding: utf-8 -*-

import gym
import numpy as np
import ray
import tensorflow as tf
from gym import Env

from agents.actor import Actor
from agents.leaner import Leaner
from buffers import ReplayBuffer
from config import MuZeroConfig
from storage import SharedStorage


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def muzero(config: MuZeroConfig):
    ray.init()
    storage = SharedStorage.remote(config)
    replay_buffer = ReplayBuffer.remote(config)

    leaner = Leaner.remote(config, storage, replay_buffer)

    temperatures = list(np.linspace(1.0, 0.1, num=config.num_actors))
    actors = [Actor.remote(config, storage, replay_buffer, temperature) for temperature in temperatures]
    workers = [leaner] + actors

    ray.get([worker.start.remote() for worker in workers])
    ray.shutdown()


def make_config(environment: Env) -> MuZeroConfig:
    return MuZeroConfig(
        env=environment,
        state_space_size=int(np.prod(env.observation_space.shape)),
        action_space_size=env.action_space.n,
        max_moves=500,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=15,  # Number of future moves self-simulated
        batch_size=64,
        td_steps=10,  # Number of steps in the future to take into account for calculating the target value
        num_actors=4,
        training_steps=int(1e8),  # Max number of training steps
        checkpoint_interval=10,
        save_interval=10000,
        lr_init=1e-4,
        lr_decay_steps=1000,
        lr_decay_rate=0.9)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    muzero(make_config(env))
