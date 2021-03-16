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
        res = {}
        for i in range(len(policy_logits[0])):
            res[Action(i)] = policy_logits[0][i]
        return res
        #return {Action(i): logit for i, logit in enumerate(policy_logits[0])}
