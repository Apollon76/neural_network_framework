#include <gmock/gmock.h>
#include "src/layers.hpp"

double sigmoidGradient(double x) {
    return exp(x) / pow((exp(x) + 1), 2);
}

const double eps = 1e-9;

void MATRIX_SHOULD_BE_EQUAL_TO(const arma::mat &actual, const arma::mat &expected, double tolerance = eps) {
    std::stringstream message;
    message << "Expected matrix: " << std::endl;
    arma::arma_ostream::print(message, expected, true);
    message << "But given matrix: " << std::endl;
    arma::arma_ostream::print(message, actual, true);
    ASSERT_TRUE(arma::approx_equal(actual, expected, "both", tolerance, tolerance)) << message.str();
}

TEST(SigmoidActivationLayerTest, TestPullGradientsBackward) {
    auto layer = SigmoidActivationLayer();
    auto input_batch = arma::mat(
            {
                    {0, 0, 0},
                    {0, 0, 0}
            });
    auto output_gradients = arma::mat(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = arma::mat(
            {
                    {sigmoidGradient(1), sigmoidGradient(2), sigmoidGradient(3)},
                    {sigmoidGradient(4), sigmoidGradient(5), sigmoidGradient(6)},
            }
    );
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, arma::mat(0, 0));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
}


TEST(DenseLayerTest, TestPullGradientsBackward) {
    auto layer = DenseLayer(3, 2);
    auto i = arma::mat(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            });
    auto og = arma::mat(
            {
                    {7, 8},
                    {9, 10}
            }
    );
    auto gradients = layer.PullGradientsBackward(i, og);
    auto expected_layer_gradients = arma::mat(
            {
                    // Weights
                    {
                            og.at(0, 0) * i.at(0, 0) + og.at(1, 0) * i.at(1, 0),
                            og.at(0, 1) * i.at(0, 0) + og.at(1, 1) * i.at(1, 0)
                    },
                    {
                            og.at(0, 0) * i.at(0, 1) + og.at(1, 0) * i.at(1, 1),
                            og.at(0, 1) * i.at(0, 1) + og.at(1, 1) * i.at(1, 1)
                    },
                    {
                            og.at(0, 0) * i.at(0, 2) + og.at(1, 0) * i.at(1, 2),
                            og.at(0, 1) * i.at(0, 2) + og.at(1, 1) * i.at(1, 2)
                    },
                    // Biases
                    {
                            og.at(0, 0) + og.at(1, 0),
                            og.at(0, 1) + og.at(1, 1),
                    }
            }
    );
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, expected_layer_gradients);
}

TEST(DenseLayerTest, TestApplyGradient) {
    auto layer = DenseLayer(2, 2);
    auto w = layer.GetWeightsAndBias();
    auto g = arma::mat(
            {
                    {1, 2},
                    {3, 4},
                    {5, 6}
            }
    );
    auto expected = arma::mat(
            {
                    {w.at(0, 0) + g.at(0, 0), w.at(0, 1) + g.at(0, 1)},
                    {w.at(1, 0) + g.at(1, 0), w.at(1, 1) + g.at(1, 1)},
                    {w.at(2, 0) + g.at(2, 0), w.at(2, 1) + g.at(2, 1)}
            }
    );
    layer.ApplyGradients(g);
    MATRIX_SHOULD_BE_EQUAL_TO(layer.GetWeightsAndBias(), expected);

}

TEST(DenseLayerTest, TestApply) {
    auto layer = DenseLayer(3, 2);
    auto w = layer.GetWeightsAndBias();
    auto i = arma::mat(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            }
    );
    auto expected = arma::mat(
            {
                    {
                            i.at(0, 0) * w.at(0, 0) + i.at(0, 1) * w.at(1, 0) + i.at(0, 2) * w.at(2, 0) + w.at(3, 0),
                            i.at(0, 0) * w.at(0, 1) + i.at(0, 1) * w.at(1, 1) + i.at(0, 2) * w.at(2, 1) + w.at(3, 1)
                    },
                    {
                            i.at(1, 0) * w.at(0, 0) + i.at(1, 1) * w.at(1, 0) + i.at(1, 2) * w.at(2, 0) + w.at(3, 0),
                            i.at(1, 0) * w.at(0, 1) + i.at(1, 1) * w.at(1, 1) + i.at(1, 2) * w.at(2, 1) + w.at(3, 1)
                    },
            }
    );
    auto actual = layer.Apply(i);
    MATRIX_SHOULD_BE_EQUAL_TO(actual, expected);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
