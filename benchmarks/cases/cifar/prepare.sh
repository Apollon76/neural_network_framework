#!/usr/bin/env bash

mkdir -p data
cd data
NAME=cifar-10-binary.tar.gz
if [[ ! -f "$NAME" ]]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
fi

if [[ ! -d "cifar-10-batches-bin" ]]; then
    tar -xzvf $NAME
fi
