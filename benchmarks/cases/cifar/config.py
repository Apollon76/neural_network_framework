import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential


def load_data(data_path: str) -> (np.array, np.array, np.array, np.array):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_labels_one_hot = tf.reshape(tf.one_hot(train_labels, 10), [-1, 10])
    test_labels_one_hot = tf.reshape(tf.one_hot(test_labels, 10), [-1, 10])

    return train_images, train_labels_one_hot, test_images, test_labels_one_hot


def build_model() -> Model:
    model = Sequential([
        L.Conv2D(3, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        L.MaxPooling2D((2, 2)),
        L.Conv2D(5, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        L.MaxPooling2D((2, 2)),
        L.Flatten(),
        L.Dense(100, activation='sigmoid'),
        L.Dense(10, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return model
