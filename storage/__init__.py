from config import MuZeroConfig
from models.network import Network

import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


class SharedStorage(object):

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            with tpu_strategy.scope():  # Train the model on the TPU
                return Network(self.config)

    def save_network(self, network: Network):
        self._networks[len(self._networks)] = network
