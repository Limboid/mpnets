import attr
from typing import Any, Callable

from exports import export
import glom

from mpnets.utils.misc import is_valid_glom_string
from mpnets.utils.parsing import parse


@export
@attr.s
class GraphExecutor:
    nodes: list[Callable] = attr.ib()
    connectivity: str = attr.ib(default=None)
    links: list[tuple[str, str]] = attr.ib(default=None)
    initial_state: dict[str, dict[str, Any]] = attr.ib(default=None)
    _ready = attr.ib(init=False, default=False)

    @attr.s.on_init
    def __post_init__(self):
        assert (
            self.connectivity is not None or self.links is not None
        ), 'Must provide either "connectivity" or "links"'
        assert not (
            self.connectivity is not None and self.links is not None
        ), 'Must provide either "connectivity" or "links", not both'
        if self.connectivity is not None:
            self.links = parse(self.links)
        self.reset()

    def reset(self):
        self.prev = (self.initial_state or {node: 0 for node in self.nodes}).copy()
        self.current_outputs = {}
        self._ready = True

    def step(self, no_single_kwarg=False):
        if not self._ready:
            self.reset()

        state = {
            "prev": self.prev.copy(),
            "current": {},
        }

        # Build dictionaries of arguments for each node
        all_node_args = {}
        for src, dst in self.links:
            # if src doesn't begin with "prev" or "current", add "current." to the front
            src_splits = src.split(".")
            if len(src_splits) == 0:
                raise ValueError(f"Invalid source: {src}")
            if src_splits[0] not in ["prev", "current"]:
                src = "current." + src
            # if dst doesn't have an arg name, find the next free position
            dst_splits = dst.split(".")
            if len(dst_splits) == 0:
                raise ValueError(f"Invalid destination: {dst}")
            if dst_splits[0] == "prev":
                raise ValueError(f"Invalid destination: {dst}")
            if dst_splits[0] == "current":
                dst_splits = dst_splits[1:]
            if len(dst_splits) == 1:
                # there is no arg name, so use the src name
                dst_node_name = dst_splits[0]
                dst_arg_name = src.split(".")[1]  # prev/current.node_name(.arg_name)?
            elif len(dst_splits) == 2:
                dst_node_name, dst_arg_name = dst_splits
            else:
                raise ValueError(f"Invalid destination: {dst}")
            # assign the src address to the arg tree
            glom.assign(
                obj=all_node_args,
                path=f"{dst_node_name}.{dst_arg_name}",
                val=src,
                missing=dict,
            )
            # if it happened to write to "current", move it to the top level
            glom.assign(all_node_args, "current", all_node_args)
            del all_node_args["current"]

        # Make all string numbers into ints
        all_node_args = {
            node: {
                int(k) if isinstance(k, str) and v.isdigit() else k: v
                for k, v in args.items()
            }
            for node, args in all_node_args.items()
        }
        """
        {
            'dstA': {
                0: 'srcA.0',
                1: 'srcA.1',
            },
            'dstB': {'srcB': 'srcB'},
        }
        """

        # Call as many nodes as possible using glom parsing
        changed = True
        while changed:
            changed = False
            for dst, args in all_node_args.items():
                if all(is_valid_glom_string(state, src) for src in args.values()):
                    self._call_node(state, dst, args)
                    break

        if all_node_args:
            # Some nodes still have dependencies. Try reading prev for them.
            changed = True
            while changed:
                changed = False
                for dst, args in all_node_args.items():
                    _args = {}
                    for arg_name, src in args:
                        if is_valid_glom_string(state, src):
                            _args[arg_name] = src
                        elif is_valid_glom_string(
                            state, src.replace("current.", "prev.")
                        ):
                            _args[arg_name] = src.replace("current.", "prev.")
                        else:
                            break  # this node still has dependencies that cannot be resolved
                    else:
                        self._call_node(state, dst, _args)
                        break

        if all_node_args:
            raise Exception(
                "Could not resolve all dependencies. Have you specified all the links?"
            )

        # Set prev_outputs to current_outputs
        self.prev = state["current"].copy()

    def _call_node(self, state, dst, args):
        if len(args) == 1 and 0 in args:
            # Special case for single positional argument
            posargs = [state[args[0]]]
            kwargs = {}
        else:
            # General case
            posarg_keys_and_refs = filter(
                lambda arg_name_and_src: isinstance(arg_name_and_src[0], int)
                and arg_name_and_src[0] >= 0,
                args.items(),
            )
            posarg_keys_and_refs = sorted(
                posarg_keys_and_refs, key=lambda arg_name_and_src: arg_name_and_src[0]
            )
            posargs = [glom.glom(state, src) for idx, src in posarg_keys_and_refs]

            kwarg_keys_and_refs = filter(
                lambda arg_name_and_src: isinstance(arg_name_and_src[0], str)
                and arg_name_and_src[0] != "",
                args.items(),
            )
            kwargs = {
                arg_name_and_ref[0]: glom.glom(state, arg_name_and_ref[1])
                for arg_name_and_ref in kwarg_keys_and_refs
            }
        fn = self.nodes[dst]
        state["current"][dst] = fn(*posargs, **kwargs)
