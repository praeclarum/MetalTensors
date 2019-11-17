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

def conv_def_weights():
    w, h = 67, 29
    x = layers.Conv2D(11, 5, input_shape=(h, w, 3), padding="same")
    m = Sequential([
        x,
        layers.ReLU(),
    ])
    m.summary()
    weights = x.get_weights()
    for i, w in enumerate(weights):
        print("CONV WEIGHTS {} SHAPED {}".format(i, w.shape))
        print(np.reshape(w, -1))

def conv_def_weights():
    w, h = 67, 29
    x = layers.BatchNormalization(11, 5, input_shape=(h, w, 3), padding="same")
    m = Sequential([
        x,
        layers.ReLU(),
    ])
    m.summary()
    weights = x.get_weights()
    for i, w in enumerate(weights):
        print("CONV WEIGHTS {} SHAPED {}".format(i, w.shape))
        print(np.reshape(w, -1))

batch_norm_def_weights()

conv_def_weights()

test_mnist()


