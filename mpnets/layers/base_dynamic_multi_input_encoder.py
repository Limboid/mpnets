import logging
from typing import Any, Callable
import torch.nn as nn


class BaseDynamicMultiInputEncoder:

    # public attributes
    encoders: dict[str, Callable]
    heads: dict[str, Callable]
    __pooling_fn = sum

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        encoders={},
        pooling_fn=None,
        encoder_factory: Callable[[Any], nn.Module] = None,
    ):
        super().__init__()

        assert (
            encoders or encoder_factory
        ), "Must provide either input_processors or input_processor_factory"

        self.encoders = encoders or {}

        self.__encoder_factory = encoder_factory
        self.__pooling_fn = pooling_fn or self.__pooling_fn

        for name, encoder in self.encoders.items():
            self.add_input(name, encoder)

    def __call__(self, **inputs):

        # check if any new inputs have been added
        for k, v in inputs.items():
            if k not in self.encoders:
                self.add_input(k, v)
                self.logger.info(f"Added new input {k}")

        # run inputs through input processors
        self.encoded_inputs = {
            k: encoder(inputs[k]) for k, encoder in self.encoders.items()
        }
        # pool the processed inputs
        self.x_pooled = self.__pooling_fn(self.encoded_inputs)

        # return outputs
        return self.x_pooled

    def add_input(self, k, example):
        self.encoders[k] = self.__encoder_factory(example)
