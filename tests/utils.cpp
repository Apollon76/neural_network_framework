#include <gmock/gmock.h>
#include <armadillo>
#include <src/tensor.hpp>
#include "utils.h"

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance) {
    std::stringstream message;
    message << "Expected matrix: " << std::endl << expected.ToString()
            << "Bug given matrix: " << std::endl << actual.ToString();
    ASSERT_TRUE(arma::approx_equal(actual.Values(), expected.Values(), "both", tolerance, tolerance)) << message.str();
}

template void
MATRIX_SHOULD_BE_EQUAL_TO<double>(const Tensor<double> &actual, const Tensor<double> &expected, double tolerance);