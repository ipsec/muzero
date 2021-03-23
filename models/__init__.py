from typing import Dict, List, NamedTuple

from games.game import Action


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]
