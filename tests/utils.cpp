#include <gmock/gmock.h>
#include <armadillo>
#include <src/tensor.hpp>
#include "utils.h"

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const arma::Mat<T> &actual, const arma::Mat<T> &expected, double tolerance) {
    ASSERT_TRUE(arma::approx_equal(actual, expected, "both", tolerance, tolerance))
                                << "Expected matrix: " << std::endl << expected
                                << "Bug given matrix: " << std::endl << actual
                                << "Diff (expected - actual): " << std::endl << (expected - actual);
}

template void
MATRIX_SHOULD_BE_EQUAL_TO<double>(const arma::Mat<double> &actual, const arma::Mat<double> &expected, double tolerance);

template<typename T>
void TENSOR_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance) {
    ASSERT_TRUE(actual.D == expected.D)
                                << "Expected tensor with dimensions " << FormatDimensions(expected)
                                << " but given " << FormatDimensions(actual);
    auto actual_matrices = std::vector<arma::Mat<T>>();
    auto expected_matrices = std::vector<arma::Mat<T>>();
    expected.template DiffWith<T>(actual,
                                  [&actual, &expected, tolerance](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                                      EXPECT_TRUE(
                                              arma::approx_equal(a, b, "absdiff", tolerance, tolerance) &&
                                              arma::approx_equal(a, b, "reldiff", tolerance, tolerance)
                                      )
                                                          << "Expected tensor: " << std::endl
                                                          << expected.ToString()
                                                          << "Bug given tensor: " << std::endl
                                                          << actual.ToString()
                                                          << "Diff (expected - actual): " << std::endl
                                                          << (a - b);
                                      arma::Mat<T> value = a - b;
                                      return value;
                                  });
}

template void
TENSOR_SHOULD_BE_EQUAL_TO<double>(const Tensor<double> &actual, const Tensor<double> &expected, double tolerance);
