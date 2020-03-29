#pragma once

#include <src/tensor.hpp>

const double eps = 1e-9;

template<typename T>
void TENSOR_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance = eps);

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const arma::Mat<T> &actual, const arma::Mat<T> &expected, double tolerance = eps);