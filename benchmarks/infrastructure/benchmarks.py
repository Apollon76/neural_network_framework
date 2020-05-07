#!/usr/bin/env python3

import subprocess
import json
import re
import os
import random
import string


def prepare(test_name):
    subprocess.check_call([f'cd cases/{test_name} && ./prepare.sh'], shell=True)


def build_docker(path):
    print('Building docker...', flush=True)
    output = subprocess.check_output([f'docker build {path}'], shell=True)
    image_id = re.findall('(?<=Successfully built )(\\w+)$', output.decode())[0]

    print(f'Image id: {image_id}', flush=True)
    return image_id


# Do it manually because on MacOS tempfile.mkdtemp() creates dir on path, which can't be mounted by docker
def mktempdir():
    rand_name = ''.join([random.choice(string.ascii_letters) for x in range(20)])
    path = f'/tmp/tmp-{rand_name}'
    os.makedirs(path)
    return path


def mktempfile():
    path = os.path.join(mktempdir(), 'file')
    open(path, 'w').close()
    return path


def run_nn_framework(test_name, data_path, epochs, batch_size):
    repo_path = os.path.dirname(os.getcwd())
    results_file = mktempfile()
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
    results_file = mktempfile()
    keras_cache_path = '/tmp/keras-cache'
    os.makedirs(keras_cache_path, exist_ok=True)

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
        '-v'
        f'"{keras_cache_path}:/home/user/.keras"',
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
    results_dir = mktempdir()
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

    tests = {
        ['mnist', 32, 10],
        ['cifar', 128, 5]
    }
    for test_name, batch_size, epochs in tests:
        data_path = f'cases/{test_name}/data'

        prepare(test_name)

        keras_result = run_keras(test_name, data_path, epochs, batch_size)
        nn_framework_result = run_nn_framework(test_name, data_path, epochs, batch_size)

        plot_results(test_name, keras_result, nn_framework_result)

    print('Everything done')


if __name__ == '__main__':
    main()
