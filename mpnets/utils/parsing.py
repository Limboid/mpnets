import itertools
from exports import export
from pyparsing import *

from mpnets.utils.misc import unsqueeze_list

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


def ensure_scope(node, scope):
    splits = node.split(".")
    if len(splits) == 0:
        raise ValueError(f"Invalid node name: {node}")
    if splits[0] in ["prev", "current"]:
        return node
    if len(splits) == 1:
        return f"{scope}.{splits[0]}"
    if len(splits) == 2:
        return f"{scope}.{splits[0]}.{splits[1]}"
    raise ValueError(f"Invalid node name: {node}")


NODE_REF = str | tuple[str, ...] | list[str] | set[str]
EDGE_CHAIN = list[NODE_REF] | tuple[NODE_REF, ...]
ADJACENCY_LIST = list[EDGE_CHAIN] | tuple[EDGE_CHAIN, ...]


def extract_adjacency_list(parsed_doc) -> ADJACENCY_LIST:
    adjacency_list = []
    for line in parsed_doc:
        if "edges" in line:
            edge_chain = []
            for i, (srcs, dsts) in enumerate(zip(line["edges"][0:], line["edges"][1:])):
                srcs = unsqueeze_list(srcs)
                dsts = unsqueeze_list(dsts)
                if i == 0:
                    edge_chain.append(srcs)
                edge_chain.append(dsts)
            adjacency_list.append(edge_chain)
    return adjacency_list


## def extract_links(parsed_doc) -> ADJACENCY_LIST:
##     links = []
##     for line in parsed_doc:
##         if "edges" in line:
##             for i, (srcs, dsts) in enumerate(zip(line["edges"][0:], line ["edges"][1:])):
##                 default_scope = "prev" if i == 0 else "current"
##                 for src, dst in itertools.product(srcs, dsts):
##                     src = ensure_scope(src, default_scope)
##                     dst = ensure_scope(dst, default_scope)
##                     links.append((src, dst))
##     return links


@export
def parse(text=None, path=None):
    assert not (
        text is None and path is None
    ), "Must provide either text or path, not both"
    assert not (text is not None and path is not None), "Must provide either text or path"

    if path is not None:
        with open(path, "r") as f:
            text = f.read()

    parse_tree = document.parseString(text)
    adjacency_list = extract_adjacency_list(parse_tree)
    return adjacency_list
