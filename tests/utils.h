#pragma once

const double eps = 1e-9;

void MATRIX_SHOULD_BE_EQUAL_TO(const arma::mat& actual, const arma::mat& expected, double tolerance = eps);