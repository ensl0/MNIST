from tensorflow import keras
import numpy as np
from keras.datasets import mnist

class Dataset():
    trainx = None
    trainy = None

    testx = None
    testy = None

    def __init__(self):
        train, test = mnist.load_data(path="mnist.npz")

        self.trainx = train[0].astype("float32") / 255
        self.trainy = keras.utils.to_categorical(train[1], 10)
        self.testx = test[0].astype("float32") / 255
        self.testy = keras.utils.to_categorical(test[1], 10)
