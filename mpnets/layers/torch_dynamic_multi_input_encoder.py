from typing import Any, Callable
from mpnets.layers.base_dynamic_multi_input_encoder import BaseDynamicMultiInputEncoder
import torch
import torch.nn as nn
import torch.optim as optim


class TorchDynamicMultiInputEncoder(nn.Module, BaseDynamicMultiInputEncoder):

    # attrs
    __pooling_fn = torch.mean
    optimizer: optim.Optimizer

    def __init__(
        self,
        encoders={},
        pooling_fn=None,
        encoder_factory: Callable[[Any], nn.Module] = None,
    ):
        super().__init__(
            encoders=encoders,
            pooling_fn=pooling_fn or self.__pooling_fn,
            encoder_factory=encoder_factory,
        )

        # overrite base class dict
        self.encoders = torch.ModuleDict(self.encoders)

    def add_input(self, k, example):
        self.encoders[k] = self.__encoder_factory(example)
