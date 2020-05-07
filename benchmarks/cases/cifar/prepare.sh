#!/usr/bin/env bash

mkdir -p data
cd data

if [[ ! -d "cifar-10-batches-bin" ]]; then

    NAME=cifar-10-binary.tar.gz
    if [[ ! -f "$NAME" ]]; then
        wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    fi

    tar -xzvf $NAME
fi
