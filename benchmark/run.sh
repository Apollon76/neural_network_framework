echo "Running benchmark"
set -xe

# Run our nn
BUILD_DIR=/nn_framework_bin
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake /nn_framework/benchmark
make
time ./benchmark -d Test_path

# Run keras
python3 /nn_framework/benchmark/prepare_data.py\
  --path /nn_framework/data/kaggle-digit-recognizer/train.csv\
  --dest /nn_framework/benchmark/test_data
time python3 /nn_framework/benchmark/learn_dense.py --path /nn_framework/benchmark/test_data
rm -rf /nn_framework/benchmark/test_data