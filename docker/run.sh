#!/bin/bash

set -xe

BUILD_DIR=/nn_framework_bin
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake /nn_framework
make
./src/main
