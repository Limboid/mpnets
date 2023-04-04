# %%
from pyparsing import *

# utils
whitespace_until_end_of_line = Regex(r"[\s]*$")

# define grammar for node and property names
varname = alphas + nums + "_"
varnames = Word(varname)
propname = Literal(".") + varnames
single_node = Combine(varnames + ZeroOrMore(propname))

# define grammar for nodes and groups of nodes
_node_list = delimitedList(single_node)
node_list = Group(
    _node_list | (Literal("(").suppress() + _node_list + Literal(")").suppress())
).setResultsName("nodes")

# define grammar for edges
edge_chain = Group(
    node_list + OneOrMore(Literal("-->").suppress() + node_list)
).setResultsName("edges")

# define grammar for graph
comment = Literal("#") + restOfLine
line = Group(
    (
        edge_chain
        | comment
        # | (edge_chain + whitespace_until_end_of_line)
        # | comment
        # | whitespace_until_end_of_line
    )
)

document = ZeroOrMore(line) + StringEnd()


def extract_links(parsed_doc):
    links = []
    for line in parsed_doc:
        if "edges" in line:
            nodes_list = line["edges"][0]
            dest_list = line["edges"][1]
            for node in nodes_list:
                for dest in dest_list:
                    links.append((node, dest))
    return links


graph_str = """
# this is a comment
node1 --> node2
node2.out1 --> node3
node2.out2 --> node4
node3 --> node4.0
node4 --> node2
(node1, node2, node3) --> node5 --> (node6, node7.input1)
node5 --> node7.input2
""".strip()

parsed_doc = document.parseString(graph_str)
links = extract_links(parsed_doc)
links

# %%
def sum_node(*args, **kwargs):
    return sum(args + tuple(kwargs.values()))


sum_nodes = {f"node{i}": sum_node for i in range(0, 3 + 1)}


def node4(node1, node2):
    return node1 + node2


def node5(node1, node2):
    return {"out1": node1, "out2": node1 + node2}


nodes = sum_nodes.copy()
nodes.update({"node4": node4, "node5": node5})

graph = """
prev.node0 --> node1
prev.node1 --> node2
prev.node2 --> node3
prev.node1 --> node4
prev.node2 --> node4
prev.node1 --> node5
prev.node2 --> node5
node5.out1 --> node0.recurrent_input
node5.out2 --> node1.recurrent_input
""".strip()

links = extract_links(document.parseString(graph))
nodes, links

######## # %%
######## def sum_node(*args, **kwargs):
########     return sum(args + tuple(kwargs.values()))
########
########
######## nodes = {f"node{i}": sum_node for i in range(0, 3 + 1)}
########
######## graph = """
######## prev.node0 --> node1
######## node1 --> node2
######## node2 --> node3
######## node3 --> node0
######## """.strip()
########
######## links = extract_links(document.parseString(graph))
######## nodes, links

# %%
import glom


def is_valid_glom_string(obj, glom_str):
    try:
        glom.glom(obj, glom_str)
        return True
    except:
        return False


# %%
import attr
from typing import Any, Callable


@attr.s
class NodeRunner:
    nodes: list[Callable] = attr.ib()
    links: list[tuple[str, str]] = attr.ib()
    initial_state: dict[str, dict[str, Any]] = attr.ib(default=None)
    _ready = attr.ib(init=False, default=False)

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
                    # All arguments are available, call the node
                    if no_single_kwarg and len(args) == 1 and 0 in args:
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
                            posarg_keys_and_refs,
                            key=lambda arg_name_and_src: arg_name_and_src[0],
                        )
                        posargs = [
                            glom.glom(state, src) for idx, src in posarg_keys_and_refs
                        ]

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
                    changed = True
                    all_node_args.pop(dst)
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
                        args = _args
                        # All arguments are available, call the node
                        if no_single_kwarg and len(args) == 1 and 0 in args:
                            # Special case for single positional argument
                            posargs = [state[args[0]]]
                            kwargs = {}
                        else:
                            # General case
                            posarg_keys_and_refs = filter(
                                lambda arg_name_and_src: isinstance(
                                    arg_name_and_src[0], int
                                )
                                and arg_name_and_src[0] >= 0,
                                args.items(),
                            )
                            posarg_keys_and_refs = sorted(
                                posarg_keys_and_refs,
                                key=lambda arg_name_and_src: arg_name_and_src[0],
                            )
                            posargs = [
                                glom.glom(state, src) for idx, src in posarg_keys_and_refs
                            ]

                            kwarg_keys_and_refs = filter(
                                lambda arg_name_and_src: isinstance(
                                    arg_name_and_src[0], str
                                )
                                and arg_name_and_src[0] != "",
                                args.items(),
                            )
                            kwargs = {
                                arg_name_and_ref[0]: glom.glom(state, arg_name_and_ref[1])
                                for arg_name_and_ref in kwarg_keys_and_refs.items()
                            }
                        fn = self.nodes[dst]
                        state["current"][dst] = fn(*posargs, **kwargs)
                        changed = True
                        all_node_args.pop(dst)
                        break

        if all_node_args:
            raise Exception(
                "Could not resolve all dependencies. Have you specified all the links?"
            )

        # Set prev_outputs to current_outputs
        self.prev = state["current"].copy()


node_runner = NodeRunner(nodes, links)
node_runner.step()
