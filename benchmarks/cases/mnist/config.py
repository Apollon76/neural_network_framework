import os

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as L


def load_data(data_path: str) -> (np.array, np.array, np.array, np.array):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None)
    test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=None)

    def prepare_data(data):
        X = data.iloc[:, 1:].values.astype('float32') / 255
        y = pd.get_dummies(data.iloc[:, 0]).values
        return X, y

    X_train, y_train = prepare_data(train)
    X_test, y_test = prepare_data(test)

    return X_train, y_train, X_test, y_test


def build_model() -> Model:
    model = Sequential([
        L.Input(784),
        L.Dense(100, activation='sigmoid'),
        L.Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model
