#include <gmock/gmock.h>
#include "src/layers.hpp"
#include "src/neural_network.hpp"
#include "src/optimizer.hpp"
#include "src/utils.hpp"

double sigmoidGradient(double x) {
    return exp(-x) / pow((exp(-x) + 1), 2);
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
                    {1, 2, 3},
                    {4, 5, 6}
            });
    auto output_gradients = arma::mat(
            {
                    {10, 20, 30},
                    {40, 50, 60}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = arma::mat(
            {
                    {10 * sigmoidGradient(1), 20 * sigmoidGradient(2), 30 * sigmoidGradient(3)},
                    {40 * sigmoidGradient(4), 50 * sigmoidGradient(5), 60 * sigmoidGradient(6)},
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

TEST(NeuralNetworkTest, TestLinearDependency) {
    auto network = NeuralNetwork(std::make_unique<Optimizer>(0.01), std::make_unique<MSELoss>());
    network.AddLayer(std::make_unique<DenseLayer>(1, 1));
    auto inputs = CreateMatrix(
            {
                    {1},
                    {5}
            });
    auto outputs = CreateMatrix(
            {
                    {2 * 1 + 3},
                    {2 * 5 + 3}
            });
    for (int i = 0; i < 10000; i++) {
        network.Fit(inputs, outputs);
    }
    MATRIX_SHOULD_BE_EQUAL_TO(
            dynamic_cast<DenseLayer *>(network.GetLayer(0))->GetWeightsAndBias(),
            CreateMatrix(
                    {
                            {2},
                            {3}
                    }),
            1e-5);
}

TEST(NeuralNetworkTest, TestLinearDependencyWithSigmoid) {
    auto network = NeuralNetwork(std::make_unique<Optimizer>(0.1), std::make_unique<MSELoss>());
    network.AddLayer(std::make_unique<DenseLayer>(1, 1));
    network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    auto inputs = CreateMatrix(
            {
                    {-2},
                    {-1},
                    {0},
                    {1},
                    {2},
            });
    auto outputs = CreateMatrix(
            {
                    {1.0 / (exp(-(2 * (-2) + 3)) + 1)},
                    {1.0 / (exp(-(2 * (-1) + 3)) + 1)},
                    {1.0 / (exp(-(2 * 0 + 3)) + 1)},
                    {1.0 / (exp(-(2 * 1 + 3)) + 1)},
                    {1.0 / (exp(-(2 * 2 + 3)) + 1)},
            });
    for (int i = 0; i < 10000; i++) {
        network.Fit(inputs, outputs);
    }
    MATRIX_SHOULD_BE_EQUAL_TO(
            dynamic_cast<DenseLayer *>(network.GetLayer(0))->GetWeightsAndBias(),
            CreateMatrix(
                    {
                            {2},
                            {3}
                    }),
            1e-1);
}

TEST(MSETest, TestLoss) {
    auto loss = MSELoss();
    ASSERT_DOUBLE_EQ(loss.GetLoss(CreateMatrix({{1, 2, 3}}), CreateMatrix({{5, 9, -1}})),
                     pow(1 - 5, 2) + pow(2 - 9, 2) + pow(3 + 1, 2));
}

TEST(MSETest, TestDerivative) {
    auto loss = MSELoss();
    auto gradients = loss.GetGradients(CreateMatrix({{1, 2, 3}}), CreateMatrix({{5, 9, -1}}));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients, CreateMatrix({{2 * (1 - 5), 2 * (2 - 9), 2 * (3 + 1)}}));
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
