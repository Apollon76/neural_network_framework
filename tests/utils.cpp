#include <gmock/gmock.h>
#include <armadillo>
#include "utils.h"

void MATRIX_SHOULD_BE_EQUAL_TO(const arma::mat& actual, const arma::mat& expected, double tolerance) {
    std::stringstream message;
    message << "Expected matrix: " << std::endl;
    arma::arma_ostream::print(message, expected, true);
    message << "But given matrix: " << std::endl;
    arma::arma_ostream::print(message, actual, true);
    ASSERT_TRUE(arma::approx_equal(actual, expected, "both", tolerance, tolerance)) << message.str();
}
