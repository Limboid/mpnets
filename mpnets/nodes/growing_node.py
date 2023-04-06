import logging
from typing import Callable
from mpnets.nodes.node import Node
from mpnets.utils.misc import error
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class GrowingCell(nn.Module, Node):

    # hparams
    lr = 0.01
    momentum = 0.9

    # public attributes
    input_processors: dict[str, Callable]
    heads: dict[str, Callable]
    __pooling_fn = torch.mean
    optimizer: optim.Optimizer = None

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        input_processors={},
        heads={},
        pooling_fn=None,
        neck_fn=None,
        optimizer: optim.Optimizer = None,
        input_processor_factory: Callable[[int], nn.Module] = None,
        head_factory: Callable[[int], nn.Module] = None,
        **hparams,
    ):
        super().__init__()

        assert (
            input_processors or input_processor_factory
        ), "Must provide either input_processors or input_processor_factory"
        assert heads or head_factory, "Must provide either heads or head_factory"

        self.input_processors = input_processors or {}
        self.heads = heads or {}

        self.__pooling_fn = pooling_fn or self.__pooling_fn
        self.__neck_fn = neck_fn or (lambda x: x)
        self.__input_processor_factory = input_processor_factory
        self.__head_factory = head_factory

        self.__dict__.update(hparams)

        self.optimizer = (
            optimizer
            or self.optimizer
            or optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        )

        for name, input_processor in self.input_processors.items():
            self.add_input(name, input_processor)
        for name, head in self.heads.items():
            self.add_head(name, head)

    def forward(self, **inputs: dict[str, Tensor]) -> dict[str, Tensor]:

        # check if any new inputs have been added
        if any(name not in self.input_processors for name in inputs):
            existing_processors = set(self.input_processors.keys())
            input_names = set(inputs.keys())
            for name in input_names - existing_processors:
                self.add_input(name, inputs[name].shape[1])
                self.logger.info(f"Added new input {name}")

        # run inputs through input processors
        self.processed_inputs = {
            name: input_processor(inputs[name])
            for name, input_processor in self.input_processors.items()
        }
        # pool the processed inputs
        self.x_pooled = self.__pooling_fn(self.processed_inputs)
        # run pooled inputs through neck
        self.x_necked = self.__neck_fn(self.x_pooled)
        # run pooled inputs through heads
        self.outputs = {name: head(self.x_necked) for name, head in self.heads.items()}

        # return outputs
        return self.outputs

    def add_input(self, name, input_processor):
        self._add_input(name, input_processor)

    def add_head(self, name, head):
        self._head_factory(name, head)

    def _add_input(self, name, size):
        self.input_processors[name] = self.__input_processor_factory(size)
        self.optimizer.add_param_group(
            {
                "params": self.input_processors[name].parameters(),
            }
        )

    def _head_factory(self, name, size):
        self.heads[name] = self.__head_factory(size)
        self.optimizer.add_param_group(
            {
                "params": self.heads[name].parameters(),
            }
        )
