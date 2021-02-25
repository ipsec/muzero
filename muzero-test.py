import operator

import gym

from games.game import make_atari_config
from muzero import load_checkpoints
from storage import SharedStorage

env = gym.make("LunarLander-v2")
observation = env.reset()

config = make_atari_config()
storage = SharedStorage(config)
network = storage.latest_network()
load_checkpoints(network)


def get_action(policy):
    return max(policy.items(), key=operator.itemgetter(1))[0].index


for _ in range(1000):
    env.render()
    network_output = network.initial_inference(observation)
    action = get_action(network_output.policy_logits)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
