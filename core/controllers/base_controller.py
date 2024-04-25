import numpy as np
from typing import Dict, List


class BaseController:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def predict_action(self, obs_dict: Dict[str, List]) -> np.ndarray:
        raise NotImplementedError()

    def reset(self):
        # reset state for stateful policies
        pass
