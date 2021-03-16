import tensorflow as tf
from pathlib import Path

from models.network import Network


def save_checkpoints(network: Network):
    try:
        for model in network.get_networks():
            Path.mkdir(Path(f'./checkpoints/muzero/{model.__class__.__name__}'), parents=True, exist_ok=True)
            model.save_weights(f'./checkpoints/muzero/{model.__class__.__name__}/checkpoint')
    except Exception as e:
        print(f"Unable to save networks. {e}")


def load_checkpoints(network: Network):
    try:
        for model in network.get_networks():
            path = Path(f'./checkpoints/muzero/{model.__class__.__name__}/checkpoint')
            if Path.exists(path):
                model.load_weights(path)
                print(f"Load weights with success.")
    except Exception as e:
        print(f"Unable to load networks. {e}")


def export_models(network: Network):
    try:
        for model in network.get_networks():
            path = Path(f'./data/saved_model/muzero/{model.__class__.__name__}')
            Path.mkdir(path, parents=True, exist_ok=True)
            tf.saved_model.save(model, str(path.absolute()))
            # print(f"Model {model.__class__.__name__} Saved!")
    except Exception as e:
        print(f"Unable to save networks. {e}")
