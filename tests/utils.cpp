#include <gmock/gmock.h>
#include <armadillo>
#include <src/tensor.hpp>
#include "utils.h"

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance) {
    ASSERT_TRUE(actual.D == expected.D)
                                << "Expected tensor with dimensions " << FormatDimensions(expected)
                                << " but given " << FormatDimensions(actual);
    auto actual_matrices = std::vector<arma::Mat<T>>();
    auto expected_matrices = std::vector<arma::Mat<T>>();
    actual.Field().for_each([&actual_matrices](const arma::Mat<T> &v) { actual_matrices.push_back(v); });
    expected.Field().for_each([&expected_matrices](const arma::Mat<T> &v) { expected_matrices.push_back(v); });
    auto diff = expected;
    int id = 0;
    diff.Field().for_each([&actual_matrices, &id](arma::Mat<T> &v) { v -= actual_matrices[id++]; });
    for (size_t i = 0; i < actual_matrices.size(); i++) {
        ASSERT_TRUE(arma::approx_equal(actual_matrices[i], expected_matrices[i], "both", tolerance, tolerance))
                                    << "Expected tensor: " << std::endl << expected.ToString()
                                    << "Bug given tensor: " << std::endl << actual.ToString()
                                    << "Diff (expected - actual): " << std::endl << diff.ToString();
    }
}

template void
MATRIX_SHOULD_BE_EQUAL_TO<double>(const Tensor<double> &actual, const Tensor<double> &expected, double tolerance);