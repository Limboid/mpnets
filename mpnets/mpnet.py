from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .types import Tensor, TensorSpec
from .backend import BACKEND, Backend
from mpnets import backend


class Connection(dataclass):

    srcs: List[str]
    dsts: List[str]
    fn: Callable[[List[Tensor]], List[Tensor]]

    @property
    def trainable_params(self) -> List[Tensor]:
        if hasattr(self.fn, 'trainable_params'):
            return self.fn.trainable_params
        else:
            return []


class MPNet(dataclass):

    node_specs: Dict[str, TensorSpec]  # recurrent state storage
    # functions which may include trainable parameters
    connections: List[Connection]

    connection_optimizer: Optimizer
    parameter_optimizer: Optimizer

    built: bool = False
    vals: Dict[str, Tensor] = {}
    grads: Dict[str, Tensor] = {}

    @property
    def trainable_params(self) -> List[Tensor]:
        params = []
        for connection in self.connections:
            params.extend(connection.trainable_params)
        return params

    def build(self):
        for name, spec in self.node_specs.items():
            self.vals[name] = Tensor.zeros(spec.shape, spec.dtype)
            self.grads[name] = Tensor.zeros(spec.shape, spec.dtype)

    def update(self):
        if not self.built:
            self.build()
            self.built = True

        if backend.BACKEND == backend.Backend.tf:
            prev_vals = self.vals.copy()
            trainable_params = self.trainable_params
            with tf.GradientTape() as tape:
                self.forward()
            # grads through all connections
            connection_grads, param_grads = tape.gradient(
                self.vals, (prev_vals, trainable_params))
            for name, grad in connection_grads.items():
                self.grads[name] += grad
            # grads feeding into weights

        elif backend.BACKEND == backend.Backend.torch:
            TODO()
        elif backend.BACKEND == backend.Backend.np:
            raise NotImplementedError(
                'np backend does not support gradient propagation')
        else:
            raise NotImplementedError(f'unknown backend {BACKEND}')

    def forward():
        for connection in self.connections:
            srcs = [getattr(self, src) for src in connection.srcs]
            dst = getattr(self, connection.dsts[0])
            dst.data = connection.fn(srcs)

    def __getattribute__(self, __name: str) -> Any:
        try:
            super().__getattribute__(__name)
        except AttributeError:
            if __name in self.vals:
                return self.vals[__name]
        else:
            raise AttributeError(f"{__name} is not a valid attribute")

# TODO: just use a single framework (like torch for developer convenience)
