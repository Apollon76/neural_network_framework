#!/bin/bash

set -xe

case "$1" in
"run")
    BUILD_DIR=/nn_framework_bin
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
    cd $BUILD_DIR
    cmake /nn_framework -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
	  make
    ./src/main "${@:2}"
    ;;
"test")
    BUILD_DIR=/nn_framework_bin
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
    cd $BUILD_DIR
    cmake /nn_framework -DCMAKE_BUILD_TYPE=Release
	  make
	  ./tests/unit_tests
    ;;
"sshd")
    /usr/sbin/sshd -D
    ;;
"example-hdf5")
    exec "/nn_framework/examples/hdf5/run.sh"
    ;;
"example-digit-recognizer")
    exec "/nn_framework/examples/digit_recognizer/run.sh"
    ;;
"example-images-loading")
    exec "/nn_framework/examples/images_loading/run.sh"
    ;;
"example-interactivity")
    exec "/nn_framework/examples/interactivity/run.sh"
    ;;
"example-simple")
    exec "/nn_framework/examples/simple/run.sh"
    ;;
*)
    exec "$@"
    ;;
esac


