#!/bin/bash

set -e

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
make

cd /nn_framework/examples/simple

GLOG_logtostderr=1 $BUILD_DIR/examples/simple
