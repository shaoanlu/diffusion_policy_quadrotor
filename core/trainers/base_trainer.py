from abc import abstractmethod


class BaseDiffusionPolicyTrainer:
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def run(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str):
        raise NotImplementedError
