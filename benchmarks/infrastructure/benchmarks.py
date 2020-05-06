#!/usr/bin/env python3

import subprocess
import tempfile
import json
import re
import os


def prepare(test_name):
    subprocess.check_call([f'cd cases/{test_name} && ./prepare.sh'], shell=True)


def build_docker(path):
    print('Building docker...', flush=True)
    output = subprocess.check_output([f'docker build {path}'], shell=True)
    image_id = re.findall('(?<=Successfully built )(\\w+)$', output.decode())[0]

    print(f'Image id: {image_id}', flush=True)
    return image_id


def run_nn_framework(test_name, data_path, epochs, batch_size):
    repo_path = os.path.dirname(os.getcwd())
    results_file = tempfile.mkstemp()[1]
    build_cache_path = '/tmp/nn-framework-benchmarks-build-cache'

    image_id = build_docker('../docker')

    inner_commands = [
        'mkdir -p /tmp/nn_framework_build',
        'cd /tmp/nn_framework_build',
        'cmake -DCMAKE_BUILD_TYPE=Release /tmp/nn_framework',
        'make',
        'cd /tmp/nn_framework/benchmarks',
        f'/tmp/nn_framework_build/benchmarks/runner --test-name {test_name} --data-path {data_path} --epochs {epochs} --batch-size {batch_size}',
    ]

    inner_commands_joined = ' && '.join(inner_commands)

    command = [
        'docker',
        'run',
        '--rm',
        '-v',
        f'"{repo_path}:/tmp/nn_framework/"',
        '-v',
        f'"{build_cache_path}:/tmp/nn_framework_build/"',
        '-v',
        f'"{results_file}:/tmp/results.json"',
        f'{image_id}',
        '/bin/bash',
        '-c',
        f'"{inner_commands_joined}"',
    ]

    print(f'Running docker...')

    subprocess.check_call(' '.join(command), shell=True)

    with open(results_file) as f:
        return json.loads(f.read())


def run_keras(test_name, data_path, epochs, batch_size):
    repo_path = os.path.dirname(os.getcwd())
    results_file = tempfile.mkstemp()[1]

    image_id = build_docker('tf-docker')

    inner_commands = [
        'cd /tmp/nn_framework/benchmarks',
        f'./infrastructure/runner.py --test-name {test_name} --data-path {data_path} --epochs {epochs} --batch-size {batch_size}',
    ]

    inner_commands_joined = ' && '.join(inner_commands)

    command = [
        'docker',
        'run',
        '--rm',
        '-v',
        f'"{repo_path}:/tmp/nn_framework/"',
        '-v',
        f'"{results_file}:/tmp/results.json"',
        f'{image_id}',
        '/bin/bash',
        '-c',
        f'"{inner_commands_joined}"',
    ]

    print(f'Running docker...')

    subprocess.check_call(' '.join(command), shell=True)

    with open(results_file) as f:
        return json.loads(f.read())


def plot_results(test_name, keras_result, nn_framework_result):
    repo_path = os.path.dirname(os.getcwd())
    results_dir = tempfile.mkdtemp()
    images_dir = os.path.join(os.getcwd(), 'benchmark-results', test_name)
    os.makedirs(images_dir, exist_ok=True)

    with open(f'{results_dir}/keras.json', 'w') as f:
        json.dump(keras_result, f)

    with open(f'{results_dir}/nn_framework.json', 'w') as f:
        json.dump(nn_framework_result, f)

    image_id = build_docker('tf-docker')

    command = [
        'docker',
        'run',
        '--rm',
        '-v',
        f'"{repo_path}:/tmp/nn_framework/"',
        '-v',
        f'"{results_dir}:/tmp/results"',
        '-v',
        f'"{images_dir}:/tmp/images"',
        f'{image_id}',
        '/bin/bash',
        '-c',
        '/tmp/nn_framework/benchmarks/infrastructure/plot_results.py',
    ]

    print(f'Running docker...')

    subprocess.check_call(' '.join(command), shell=True)

    print(f'Results written to {images_dir}')


def main():
    if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) != os.getcwd():
        print('Script should be called in ./benchmarks directory')
        exit(1)

    test_name = 'mnist'
    data_path = 'cases/mnist/data'
    epochs = 10
    batch_size = 32

    prepare(test_name)

    keras_result = run_keras(test_name, data_path, epochs, batch_size)
    nn_framework_result = run_nn_framework(test_name, data_path, epochs, batch_size)

    plot_results(test_name, keras_result, nn_framework_result)

    print('Everything done')


if __name__ == '__main__':
    main()
