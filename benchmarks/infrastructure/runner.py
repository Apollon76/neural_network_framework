#!/usr/bin/env python3

import os
import time
import contextlib
import datetime
import argparse
import json
import importlib.util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@contextlib.contextmanager
def Action(name, endl=False):
    print(f'{name}...', end='\n' if endl else '', flush=True)
    start = time.time()
    try:
        yield None
    except Exception:
        print('Failed ', end='', flush=True)
        raise
    else:
        print('Done ', end='', flush=True)
    finally:
        print(f' ({datetime.timedelta(0, time.time() - start)}s)', flush=True)


def load_benchmark_configurator(path):
    spec = importlib.util.spec_from_file_location("", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description='Run model fitting and evaluation')
    parser.add_argument('-t', '--test-name', type=str, required=True, help='name of benchmark to run (./cases/*)')
    parser.add_argument('-d', '--data-path', type=str, required=True, help='path to data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs to fit')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='batch size used while fitting')
    args = parser.parse_args()

    config_path = f'./cases/{args.test_name}/config.py'
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size

    with Action(f'Loading configuration from path "{config_path}"'):
        benchmark_configurator = load_benchmark_configurator(config_path)

    with Action(f'Loading data from {data_path}'):
        X_train, y_train, X_test, y_test = benchmark_configurator.load_data(data_path)

    with Action(f'Building model'):
        model = benchmark_configurator.build_model()

    with Action(f'Fitting model for {epochs} epochs', endl=True):
        start = time.time()
        hist = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size)
        total_time_sec = time.time() - start

    with Action(f'Evaluating model on test data'):
        results = model.evaluate(X_test, y_test, verbose=0)

    eval_res = '\n'.join([f'{name}: {res}' for name, res in zip(model.metrics_names, results)])
    print(f'Evaluation result:\n{eval_res}')

    result = {
        'epochs': epochs,
        'batch_size': batch_size,
        'time_ms_total': int(total_time_sec * 1000),
        'metrics': {}
    }

    for name, values in hist.history.items():
        subset = 'train'
        if name.startswith('val_'):
            name = name[4:]
            subset = 'test'
        if name not in result['metrics']:
            result['metrics'][name] = {}
        result['metrics'][name][subset] = [(i, float(x)) for i, x in enumerate(values)]

    with open('/tmp/results.json', 'w') as f:
        f.write(json.dumps(result))


if __name__ == '__main__':
    main()
