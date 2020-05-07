#!/bin/bash

set -e

cd ./../../data/mnist-png
./unpack.sh

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
make

cd /nn_framework/examples/images_loading

$BUILD_DIR/examples/images_loading --data-path /nn_framework/data
