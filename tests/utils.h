#pragma once

#include <src/tensor.hpp>

const double eps = 1e-9;

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance = eps);