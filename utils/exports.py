import tensorflow as tf

from pathlib import Path
from models.network import Network


def export_models(network: Network):
    try:
        for model in network.get_networks():
            path = Path(f'./data/saved_model/muzero/{model.__class__.__name__}')
            Path.mkdir(path, parents=True, exist_ok=True)
            tf.saved_model.save(model, str(path.absolute()))
    except Exception as e:
        print(f"Unable to save networks. {e}")
