from config import MuZeroConfig
from models.network import Network


class SharedStorage(object):

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network(self.config)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
