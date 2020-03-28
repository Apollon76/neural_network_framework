#include <gtest/gtest.h>
#include <src/scoring/scoring.hpp>

using namespace nn_framework;

TEST(ScoringTest, TestMSEScore) {
    auto a = Tensor<double>::init({1, 2, 3});
    auto b = Tensor<double>::init({3, 2, 1});
    ASSERT_DOUBLE_EQ(nn_framework::scoring::mse_score(a, b), 8.0 / 3);
}
