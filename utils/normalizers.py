import numpy as np


class BaseNormalizer:
    def normalize_data(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize_data(self, *args, **kwargs):
        raise NotImplementedError()


class LinearNormalizer(BaseNormalizer):
    def __init__(self):
        pass

    def normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - np.array(stats["min"])) / (np.array(stats["max"]) - np.array(stats["min"]))
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (np.array(stats["max"]) - np.array(stats["min"])) + np.array(stats["min"])
        return data
