#!/bin/bash

set -xe

case "$1" in
"run")
    BUILD_DIR=/nn_framework_bin
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
    cd $BUILD_DIR
    cmake /nn_framework
	make
    ./src/main "${@:2}"
    ;;
"sshd")
    /usr/sbin/sshd -D
    ;;
*)
    exec "$@"
    ;;
esac


