import ray
from config import MuZeroConfig
from models.network import Network


@ray.remote
class SharedStorage(object):

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.network = Network(self.config)
        self._started = False

    def get_network_weights(self):
        return self.network.get_weights()

    def update_network(self, weights):
        self.network.set_weights(weights)
        if not self._started:
            self._started = True

    def started(self):
        return self._started
