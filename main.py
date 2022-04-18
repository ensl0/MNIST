from tensorflow import keras
from Dataset import Dataset
from Model import Model
from os import getcwd

def Train_Model():
    model = Model(data.trainx, data.trainy)
    model.Initialize_Model()

def Test_Model():
    model = keras.models.load_model(getcwd())
    model.evaluate(data.testx, data.testy)

if __name__ == '__main__':
    print(getcwd())
    data = Dataset()
    Train_Model()
    Test_Model()

