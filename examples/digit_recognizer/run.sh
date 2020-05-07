#!/bin/bash

set -e

cd /nn_framework/data/kaggle-digit-recognizer
./unpack.sh

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
make

cd /nn_framework/examples/digit_recognizer

$BUILD_DIR/examples/digit_recognizer --data-path /nn_framework/data --output-path /nn_framework/examples/digit_recognizer
