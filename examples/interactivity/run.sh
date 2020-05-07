#!/bin/bash

set -e

cd /nn_framework/data/kaggle-digit-recognizer
./unpack.sh

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
make

cd /nn_framework/examples/interactivity

$BUILD_DIR/examples/interactivity --data-path /nn_framework/data
