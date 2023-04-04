import coconut

import re
from typing import Any, Callable, Dict, List, Tuple, Union
from glom import glom


class RGNet:
    def __init__(
        self,
        layers: Union[List[Callable], Dict[str, Callable]],
        connectivity: Union[str, List[Tuple[str, str]]],
    ):
        if isinstance(layers, list):
            self.layers = {layer.__name__: layer for layer in layers}
        else:
            self.layers = layers

        if isinstance(connectivity, str):
            self.connectivity = self.parse_dsl(connectivity)
        else:
            self.connectivity = connectivity

        self.prev = {}

    def parse_dsl(self, dsl: str) -> List[Tuple[str, str]]:
        connections = []
        tokens = self.lex(dsl)
        while tokens:
            src, dst = self.parse_connection(tokens)
            connections.append((src, dst))
        return connections

    def lex(self, dsl: str) -> List[str]:
        # Remove comments
        dsl = re.sub(r"#.*", "", dsl)

        # Tokenize the DSL
        tokens = re.findall(r"\w+(\.\w+)?|[-<>]", dsl)
        return tokens

    def parse_connection(self, tokens: List[str]) -> Tuple[str, str]:
        src, arrow, dst = tokens[:3]
        if arrow != "->":
            raise ValueError(f"Invalid syntax: expected '->', got '{arrow}'")
        tokens.pop(0)
        tokens.pop(0)
        tokens.pop(0)
        return src, dst

    def __call__(self, *inputs, **kwargs):
        # Initialize the previous step outputs with the inputs
        for idx, input in enumerate(inputs):
            self.prev[f"input.{idx}" if idx else "input"] = input

        # Process the inputs and feed them through the network according to the connectivity
        for src, dst in self.connectivity:
            layer_name, output_idx = re.match(r"(\w+)(?:\.(\w+))?", src).groups()
            output = self.prev[layer_name]
            if output_idx:
                output = glom(output, output_idx)

            layer_name, input_idx = re.match(r"(\w+)(?:\.(\w+))?", dst).groups()
            layer = self.layers[layer_name]

            if input_idx:
                prev_inputs = self.prev.get(layer_name, {})
                prev_inputs[input_idx] = output
                self.prev[layer_name] = prev_inputs
            else:
                self.prev[layer_name] = layer(output)

        return self.prev["output"]


# Example usage
mpnet = RGNet(
    layers=[...],
    connectivity="""
        # this designs a simple multilayer RNN with indexed inputs and outputs
        input.0 -> dense0.0
        dense0.1 -> dense1.0
        dense1.1 -> dense2.0
        dense2.1 -> dense0.0
        dense2.2 -> dense3.0
        dense1.2 -> dense3.1
        dense3.2 -> output.0
    """,
)

connectivity = """
    input -> dense0
    dense0 -> dense1
    dense1 -> output
"""

connectivity = """
    input.0 -> dense0.a
    input.1 -> dense0.b
    dense0 -> dense1
    dense1 -> output
"""

connectivity = """
    input -> dense0.a
    dense0 -> dense1
    dense1.0 -> dense2.a
    dense1.1 -> dense2.b
    dense2 -> output
"""

connectivity = """
    input.0 -> dense0.a
    input.1 -> dense0.b
    dense0.0 -> dense1.a
    dense0.1 -> dense1.b
    dense1.0 -> dense2.a
    dense1.1 -> dense2.b
    dense1.2 -> dense2.c
    dense2 -> output
"""
