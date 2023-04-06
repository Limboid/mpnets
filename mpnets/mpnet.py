from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar
from mpnets.core.graph_executor import GraphExecutor
from mpnets.utils.parsing import ADJACENCY_LIST

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl


NODE = str | Callable
EDGE = tuple[NODE, NODE] | list
EDGE_CHAIN = list


class MPNet(pl.LightningModule, GraphExecutor):
    def __init__(
        self,
        nodes: list[Callable],
        connections: str = None,
        adjacency_list: ADJACENCY_LIST = None,
        **hparams
    ):
        super().__init__(
            nodes=nodes,
            connectivity=connections,
            adjacency_list=adjacency_list,
        )

    def forward(self, batch: dict[str, Tensor], **kwargs):
        pass

    def add_loss(self, name, loss):
        # losses map to `loss_nodes` in the graph
        # there are multiple, multidimensional losses with varing network coverage
        if name not in self.loss_nodes:
            self.loss_nodes[name] = loss

    def train(self, batch: dict[str, Tensor], **kwargs):
        pass

    @property
    def loss_nodes(self):
        ...

    @property
    def general_nodes(self):
        ...

    @property
    def special_nodes(self):
        ...
