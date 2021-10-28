# -*- coding: utf-8 -*-

import gym
import ray
import tensorflow as tf

from agents.actor import Actor
from agents.leaner import Leaner
from buffers import ReplayBuffer
from config import MuZeroConfig
from games.game import make_atari_config
from stats import Summary
from storage import SharedStorage


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def muzero(config: MuZeroConfig):
    ray.init()
    storage = SharedStorage.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
    summary = Summary.remote()

    leaner = Leaner.remote(config, storage, replay_buffer, summary)
    actors = [Actor.remote(config, storage, replay_buffer, summary) for _ in range(config.num_actors)]
    workers = [leaner] + actors

    ray.get([worker.start.remote() for worker in workers])
    ray.shutdown()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    muzero(make_atari_config(env))
