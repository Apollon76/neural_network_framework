import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import pandas
from sklearn.model_selection import train_test_split
import argparse

import warnings
import os
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Learn neural network.')
parser.add_argument('-p', '--path', type=str, help='Path to test data folder')
args = parser.parse_args()


def read_data(name):
    data = pandas.read_csv(os.path.join(args.path, name + '.csv'))
    X = data.iloc[:, 1:].values
    y = data["label"].values
    X = X.astype('float32')
    X = X / 255
    y = tf.keras.utils.to_categorical(y, 10)
    return X, y


X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

model = Sequential()
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

print('Result on test data: ', model.evaluate(X_test, y=y_test)[-1])
