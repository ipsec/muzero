import tensorflow as tf
from games.game import Action
from typing import Dict, List, NamedTuple


class NetworkOutput(NamedTuple):
    value: tf.Tensor
    reward: tf.Tensor
    policy_logits: Dict[Action, tf.Tensor]
    hidden_state: List[tf.Tensor]
