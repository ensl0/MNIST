import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class Model():
    dataset = None
    data = None
    labels = None

    def __init__(self, train, test):
        self.data = train
        self.labels = test

    def Initialize_Model(self):

        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(self.data, self.labels, batch_size=128, epochs=10, validation_split=0.1)

        model.save(os.getcwd())