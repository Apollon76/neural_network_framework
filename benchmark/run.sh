echo "Running benchmark"
set -xe

# Prepare data
python3 /nn_framework/benchmark/prepare_data.py\
  --path /nn_framework/data/kaggle-digit-recognizer/train.csv\
  --dest /nn_framework/benchmark/test_data

# Run our nn
BUILD_DIR=/nn_framework_bin
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake /nn_framework
make
ls ./benchmark
CPP_START=$(date +%s)
time ./benchmark/benchmark -d /nn_framework/benchmark/test_data
CPP_END=$(date +%s)
CPP_DIFF=$(( $CPP_END - $CPP_START ))

# Run keras
PY_START=$(date +%s)
time python3 /nn_framework/benchmark/learn_dense.py --path /nn_framework/benchmark/test_data
PY_END=$(date +%s)
PY_DIFF=$(( $PY_END - $PY_START ))

echo "C++ network took $CPP_DIFF seconds to learn"
echo "Python network took $CPP_DIFF seconds to learn"
rm -rf /nn_framework/benchmark/test_data