import tensorflow as tf
import pandas as pd
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def load_data(data_path):
    train = pd.read_csv(f'{data_path}/mnist/mnist_train.csv', header=None)
    test = pd.read_csv(f'{data_path}/mnist/mnist_test.csv', header=None)

    X_train = train.loc[:, 1:]
    y_train = pd.get_dummies(train[0])

    X_test = test.loc[:, 1:]
    y_test = pd.get_dummies(test[0])

    return X_train, y_train, X_test, y_test

def prepare_and_save_model(data_path, model_file):
    model = create_model()

    X_train, y_train, X_test, y_test = load_data(data_path)

    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

    model.save(model_file)

def load_and_evaluate_model(data_path, model_file):
    model = tf.keras.models.load_model(model_file)

    X_train, y_train, X_test, y_test = load_data(data_path)

    _, train_score = model.evaluate(X_train, y_train)
    _, test_score = model.evaluate(X_test, y_test)

    print(f'Train score: {train_score}, test score: {test_score}')

    assert(test_score > 0.8)

def main():
    parser = argparse.ArgumentParser(description='Keras-comparible hdf5 format example')
    parser.add_argument('--data-path', help='path to data')
    parser.add_argument('--model-file', help='file to store model')
    parser.add_argument('--mode', choices=['load', 'save'], help='mode')
    args = parser.parse_args()

    if args.mode == 'save':
        prepare_and_save_model(args.data_path, args.model_file)
    elif args.mode == 'load':
        load_and_evaluate_model(args.data_path, args.model_file)
    else:
        raise Exception(f'Unknown mode: {args.mode}')

if __name__ == '__main__':
    main()