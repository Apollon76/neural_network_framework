#include <gtest/gtest.h>
#include "src/scoring/scoring.hpp"


TEST(ScoringTest, TestMSEScore) {
    arma::vec a = {1, 2, 3};
    arma::vec b = {3, 2, 1};
    ASSERT_DOUBLE_EQ(nn_framework::scoring::mse_score(a, b), 8.0 / 3);
}
