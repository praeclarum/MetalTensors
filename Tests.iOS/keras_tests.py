#!/usr/bin/env python3

import numpy as np
import keras
from keras.models import Sequential
import keras.layers as layers

def test_mnist():
    m = Sequential([
        layers.Conv2D(32, 5, input_shape=(28, 28, 1), padding="same"),
        layers.ReLU(),
        layers.MaxPool2D(),
        layers.Conv2D(64, 5, padding="same"),
        layers.ReLU(),
        layers.MaxPool2D(),
        layers.Dense(1024),
        layers.ReLU(),
        layers.Dense(10), # Size = 7
        layers.Softmax(),
    ])
    m.summary()

test_mnist()


