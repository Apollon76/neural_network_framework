#!/bin/bash

set -e

BUILD_DIR=$(mktemp -d)

cd $BUILD_DIR || exit

cmake -DCMAKE_BUILD_TYPE=Release /nn_framework
cd $BUILD_DIR/examples/simple || exit
make

cd /nn_framework/examples/simple

GLOG_logtostderr=1 $BUILD_DIR/examples/simple/simple
