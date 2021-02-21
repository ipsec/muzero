from dataclasses import dataclass
from typing import Dict, List

from games.game import Action


@dataclass
class NetworkOutput:
    policy_logits: Dict[Action, float]
    hidden_state: List[float]
    value: float
    reward: float

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i): float(logit) for i, logit in enumerate(policy_logits[0])}
