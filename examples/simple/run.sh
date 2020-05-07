#!/bin/bash

set -e

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
make

cd /nn_framework/examples/simple

$BUILD_DIR/examples/simple --data-path /nn_framework/data --output-path /nn_framework/examples/simple