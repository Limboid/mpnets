from pyparsing import *

# define grammar for node and property names
alphanums = Word(alphanums)
propname = Literal(".").suppress() + alphanums
single_node = Combine(alphanums + ZeroOrMore(propname))

# define grammar for nodes and groups of nodes
_node_list = delimitedList(single_node)
node_list = _node_list | Group("(" + _node_list + ")")

# define grammar for edges
edge_chain = Group(node_list + OneOrMore(Literal("-->").suppress() + node_list))

# define grammar for graph
comment = Literal("#") + restOfLine
line = comment | edge_chain

# parse function
def parse_graph(s):
    # remove comments
    s = s.split("#")[0]
    # parse graph
    results = graph.parseString(s)
    # process results into adjacency list
    adjacency_list = []
    for r in results:
        if isinstance(r, str):
            continue
        elif len(r) == 2:
            adjacency_list.append((r["src"], r["dst"]))
        elif len(r) == 3:
            for n in r["nodegroup"]:
                adjacency_list.append((r["node"]["name"], n))
    return adjacency_list


graph_str = """
# example graph
node1 --> node2
node2.out1 --> node3
node2.out2 --> node4
node3 --> node4
node4 --> node2
(node1, node2, node3) --> node5 --> (node6, node7.input1)
node5 --> node7.input2
"""

adj_list = parse_graph(graph_str)
print(adj_list)
