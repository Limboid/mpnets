from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar

import torch
from torch import Tensor
import pytorch_lightning as pl


class Node(dataclass):
    size: torch.Size
    name: str = None
    pooling_function: Callable[[Tensor], Tensor] = torch.sum
    activation_function: Callable[[Tensor], Tensor] = lambda x: x
    initialization_function: Callable[[torch.Size], Tensor] = torch.zeros


class Connection(dataclass):

    srcs: List[Node]
    dsts: List[Node]
    fn: Callable[[List[Tensor]], List[Tensor]]

    @property
    def trainable_params(self) -> List[Tensor]:
        if hasattr(self.fn, 'parameters'):
            return self.fn.parameters()
        else:
            return []


class MPNet(dataclass, pl.LightningModule):

    nodes: List[Node]
    connections: List[Connection]

    # TODO  store the state optimizer in the state dict
    state_optimizer_fn: Callable[[], torch.optim.Optimizer]
    # ptl stores the model's optimizer in `.optimizer()` parameter_optimizer: torch.optim.Optimizer

    @property
    def parameters(self) -> List[Tensor]:
        return [param for connection in self.connections
                for param in connection.parameters()]

    def __init__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__()
        self.nodes = nodes
        self.connections = connections

    # TODO: implement pl.LightningModule.forward and other methods

    @property
    def _initial_state(self) -> Dict[Node, Tensor]:
        return {node: node.initialization_function(node.size)
                for node in self.nodes}

    def _update(self, vals: Dict[Node, Tensor]):

        # clear gradients

        # forward pass
        vals = self._forward(vals)

        # backward pass on state variables
        for node in self.nodes:
            # TODO
            vals[node]._grad += vals[node].backward()

        # backward pass on trainable parameters

    def _forward(self, vals: Dict[Node, Tensor]) -> Dict[Node, Tensor]:
        # compute buckets
        buckets = {node: [] for node in self.nodes}
        for connection in self.connections:
            inputs = [vals[src] for src in connection.srcs]
            outputs = connection.fn(inputs)
            for output, dst in zip(outputs, connection.dsts):
                buckets[dst].append(output)
        # pool bucket values and apply activation function
        for node in self.nodes:
            vals[node] = node.activation_function(
                node.pooling_function(buckets[node]))
        return vals
