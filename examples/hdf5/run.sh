#!/bin/bash

set -e

cd /nn_framework/data/mnist
./unpack.sh

WORKDIR=$(mktemp -d)

cd $WORKDIR || exit

cmake /nn_framework -DCMAKE_BUILD_TYPE=Release
pushd examples
make
popd

DATA_PATH=/nn_framework/data

echo ---------
echo ---------
echo Testing loading model from keras in c++
echo ---------
echo ---------

python3 /nn_framework/examples/hdf5/generate_keras_model.py --data-path $DATA_PATH --model-file $WORKDIR/mnist_model_from_keras.h5 --mode save
./examples/hdf5_example --data-path $DATA_PATH --model-file $WORKDIR/mnist_model_from_keras.h5 --mode load

echo ---------
echo ---------
echo Testing loading model from c++ in keras
echo ---------
echo ---------

./examples/hdf5_example --data-path $DATA_PATH --model-file $WORKDIR/mnist_model_from_cpp.h5 --mode save
python3 /nn_framework/examples/hdf5/generate_keras_model.py --data-path $DATA_PATH --model-file $WORKDIR/mnist_model_from_cpp.h5 --mode load