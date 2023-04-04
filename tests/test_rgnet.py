import unittest
from notebooks.rgnet import RGNet
import jax.nn as nn


class TestRGNet(unittest.TestCase):
    def test_connectivity_1(self):
        connectivity = """
            input -> dense0
            dense0 -> dense1
            dense1 -> output
        """
        layers = {
            "input": nn.Dense(20),
            "dense0": nn.Dense(30),
            "dense1": nn.Dense(1),
            "output": nn.Identity(),
        }
        rgnet = RGNet(layers=layers, connectivity=connectivity)
        # Add assertions to test rgnet object

    def test_connectivity_2(self):
        connectivity = """
            input.0 -> dense0.a
            input.1 -> dense0.b
            dense0 -> dense1
            dense1 -> output
        """
        layers = {
            "input": nn.Identity(),
            "dense0": nn.Dense(30),
            "dense1": nn.Dense(1),
            "output": nn.Identity(),
        }
        rgnet = RGNet(layers=layers, connectivity=connectivity)
        # Add assertions to test rgnet object

    def test_connectivity_3(self):
        connectivity = """
            input -> dense0.a
            dense0 -> dense1
            dense1.0 -> dense2.a
            dense1.1 -> dense2.b
            dense2 -> output
        """
        layers = {
            "input": nn.Dense(20),
            "dense0": nn.Dense(30),
            "dense1": nn.Split([10, 20], axis=1),
            "dense2": nn.Dense(1),
            "output": nn.Identity(),
        }
        rgnet = RGNet(layers=layers, connectivity=connectivity)
        # Add assertions to test rgnet object

    def test_connectivity_4(self):
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
        layers = {
            "input": nn.Identity(),
            "dense0": nn.Dense(30),
            "dense1": nn.Split([10, 10, 10], axis=1),
            "dense2": nn.Dense(1),
            "output": nn.Identity(),
        }
        rgnet = RGNet(layers=layers, connectivity=connectivity)
        # Add assertions to test rgnet object
