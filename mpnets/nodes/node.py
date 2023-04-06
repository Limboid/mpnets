from abc import abstractmethod
from enum import Enum
import attr
from exports import export


@export
@attr.s(auto_attr=True)
class Node:
    tags: list[str] = attr.ib()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def wrap(self, fn=None, *, tags=[], name=None):
        def make_custom_cell(fn):
            return type(name or fn.__name__, (Node,), {"__call__": fn, "tags": tags})

        if fn:
            return make_custom_cell(fn)
        else:
            return make_custom_cell
