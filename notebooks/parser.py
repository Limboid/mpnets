import re

# Define the tokens
WORD = "WORD"
ARROW = "ARROW"
EOL = "EOL"

# Define the regular expressions for each token type
word_regex = r"\b[a-zA-Z_]+\b"
arrow_regex = r"\s*->\s*"
eol_regex = r"[\r\n]+"

# Compile the regular expressions
word_pattern = re.compile(word_regex)
arrow_pattern = re.compile(arrow_regex)
eol_pattern = re.compile(eol_regex)

# Define a function to tokenize the input file
def tokenize(input_file):
    tokens = []
    for line in input_file:
        line = line.split("#")[0].strip()  # Remove comments
        if not line:
            continue  # Skip empty lines
        words = word_pattern.findall(line)
        arrows = arrow_pattern.findall(line)
        eols = eol_pattern.findall(line)
        for word in words:
            tokens.append((WORD, word))
        for arrow in arrows:
            tokens.append((ARROW,))
        for eol in eols:
            tokens.append((EOL,))
    return tokens


# Define the parse tree nodes
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


# Define the parser
def parse(tokens):
    nodes = []
    current_node = None
    for token in tokens:
        if token[0] == WORD:
            node = Node(token[1])
            if current_node is not None:
                current_node.add_child(node)
            current_node = node
            nodes.append(node)
        elif token[0] == ARROW:
            if current_node is None:
                raise ValueError("Unexpected token: ->")
            current_node.add_child(Node(None))
        elif token[0] == EOL:
            current_node = None
        else:
            raise ValueError("Unknown token: {}".format(token))
    return nodes


# Define a function to extract the src-dst relations from the parse tree
def extract_relations(node):
    if node.value is not None:
        relations = []
        for child in node.children:
            if child.value is not None and len(child.children) > 0:
                relations.append((node.value, child.children[0].value))
        return relations
    else:
        relations = []
        for child in node.children:
            child_relations = extract_relations(child)
            if child_relations is not None:
                relations.extend(child_relations)
        return relations


# Define a function to combine relations that share a common destination
def combine_relations(relations):
    combined = {}
    for src, dst in relations:
        if dst not in combined:
            combined[dst] = []
        combined[dst].append(src)
    return [(tuple(src_list), dst) for dst, src_list in combined.items()]


# Define a function to convert the input file to an adjacency list
def to_adjacency_list(input_file):
    tokens = tokenize(input_file)
    nodes = parse(tokens)
    relations = []
    for node in nodes:
        relations.extend(extract_relations(node))
    relations = combine_relations(relations)
    return relations


__all__ = ["to_adjacency_list"]
