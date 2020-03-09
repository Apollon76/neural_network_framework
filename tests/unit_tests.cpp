#include <gmock/gmock.h>
#include "src/layers/activations.hpp"
#include "src/layers/dense.hpp"
#include "src/neural_network.hpp"
#include "src/optimizer.hpp"
#include "src/utils.hpp"

double sigmoidGradient(double x) {
    return exp(-x) / pow((exp(-x) + 1), 2);
}

const double eps = 1e-9;

void MATRIX_SHOULD_BE_EQUAL_TO(const arma::mat& actual, const arma::mat& expected, double tolerance = eps) {
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
    auto inputs = CreateMatrix<double>(
            {
                    {1},
                    {5}
            });
    auto outputs = CreateMatrix<double>(
            {
                    {2 * 1 + 3},
                    {2 * 5 + 3}
            });
    for (int i = 0; i < 10000; i++) {
        network.Fit(inputs, outputs);
    }
    MATRIX_SHOULD_BE_EQUAL_TO(
            dynamic_cast<DenseLayer*>(network.GetLayer(0))->GetWeightsAndBias(),
            CreateMatrix<double>(
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
    auto inputs = CreateMatrix<double>(
            {
                    {-2},
                    {-1},
                    {0},
                    {1},
                    {2},
            });
    auto outputs = CreateMatrix<double>(
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
            dynamic_cast<DenseLayer*>(network.GetLayer(0))->GetWeightsAndBias(),
            CreateMatrix<double>(
                    {
                            {2},
                            {3}
                    }),
            1e-1);
}

TEST(MSETest, TestLoss) {
    auto loss = MSELoss();
    ASSERT_DOUBLE_EQ(loss.GetLoss(CreateMatrix<double>({{1, 2, 3}}), CreateMatrix<double>({{5, 9, -1}})),
                     pow(1 - 5, 2) + pow(2 - 9, 2) + pow(3 + 1, 2));
}

TEST(MSETest, TestDerivative) {
    auto loss = MSELoss();
    auto gradients = loss.GetGradients(CreateMatrix<double>({{1, 2, 3}}), CreateMatrix<double>({{5, 9, -1}}));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients, CreateMatrix<double>({{2 * (1 - 5), 2 * (2 - 9), 2 * (3 + 1)}}));
}

TEST(ReLUActivationLayerTest, TestReLULayer) {
    auto layer = ReLUActivationLayer();
    auto input_batch = arma::mat(
            {
                    {1, 2, 3},
                    {4, 5, -6}
            });
    auto actual = layer.Apply(input_batch);
    MATRIX_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), arma::mat(
            {
                    {1, 2, 3},
                    {4, 5, 0}
            }));

    auto output_gradients = arma::mat(
            {
                    {10, 20, 30},
                    {40, 50, 60}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = arma::mat(
            {
                    {10, 20, 30},
                    {40, 50, 0}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, arma::mat(0, 0));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
}

TEST(TanhActivationLayerTest, TestTanhLayer) {
    auto layer = TanhActivationLayer();
    auto input_batch = arma::mat(
            {
                    {1,  2,  3},
                    {-1, -2, -3}
            });
    auto actual = layer.Apply(input_batch);
    auto expected = arma::mat(
            {
                    {0.7615942,  0.9640276,  0.9950548},
                    {-0.7615942, -0.9640276, -0.9950548}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), expected,
                              1e-6
    );

    auto output_gradients = arma::mat(
            {
                    {10,  20,  30},
                    {-10, -20, -30}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = arma::mat(
            {
                    {4.199743,  1.4130175,  0.29598176},
                    {-4.199743, -1.4130175, -0.29598176}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, arma::mat(0, 0));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients, 1e-6);
}

TEST(MomentumOptimizerTest, TestDifferentLayers) {
    MomentumOptimizer optimizer(0.5, 0.1);
    auto firstLayer = DenseLayer(1, 1);
    auto secondLayer = DenseLayer(1, 1);

    auto firstLayerGradientStep = optimizer.GetGradientStep(arma::mat({1, 2, 3}), &firstLayer);
    auto secondLayerGradientStep = optimizer.GetGradientStep(arma::mat({10, 20, 30}), &secondLayer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstLayerGradientStep, arma::mat{-0.5, -1, -1.5});
    MATRIX_SHOULD_BE_EQUAL_TO(secondLayerGradientStep, arma::mat{-5, -10, -15});
}

TEST(MomentumOptimizerTest, TestMomentum) {
    MomentumOptimizer optimizer(0.5, 0.1);
    auto layer = DenseLayer(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(arma::mat({100, 200, 300}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(arma::mat({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(arma::mat({10, 20, 30}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep, arma::mat{-50, -100, -150});
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep, arma::mat{-10, -20, -30});
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep, arma::mat{-6, -12, -18});
}

TEST(RMSPropOptimizerTest, TestRMSPropGradientStep) {
    RMSPropOptimizer optimizer(0.5, 0.1);
    auto layer = DenseLayer(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(arma::mat({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(arma::mat({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(arma::mat({10000, 2, 3}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep, arma::mat{-0.5270463, -0.5270463, -0.5270463}, 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep, arma::mat{-0.526783  , -0.526782, -0.526782}, 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep, arma::mat{-0.5270462 , -0.1588382, -0.158838}, 1e-6);
}

TEST(AdamOptimizerTest, TestAdamGradientStep) {
    AdamOptimizer optimizer(0.5);
    auto layer = DenseLayer(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(arma::mat({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(arma::mat({-1, -2, -3}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(arma::mat({10000, 2, 3}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep, arma::mat{-15.81059779, -15.81119066, -15.81130046}, 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep, arma::mat{11.18285631, 11.18306609, 11.18310494}, 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep, arma::mat{-15.81138814, -9.133237456, -9.133258618}, 1e-6);
}

TEST(SerializationTest, TestNNSerialization) {
    auto model = NeuralNetwork(std::make_unique<Optimizer>(0.01), std::make_unique<MSELoss>());;
    model.AddLayer(std::make_unique<DenseLayer>(2, 3));
    model.AddLayer(std::make_unique<SigmoidActivationLayer>());
    model.AddLayer(std::make_unique<DenseLayer>(3, 1));
    model.AddLayer(std::make_unique<ReLUActivationLayer>());

    auto expected = R"({"layers":[{"layer_type":"dense","params":{"n_cols":3,"n_rows":2}},{"layer_type":"sigmoid_activation"},{"layer_type":"dense","params":{"n_cols":1,"n_rows":3}},{"layer_type":"relu_activation"}],"loss":["loss_type","mse"],"optimizer":{"optimizer_type":"optimizer","params":{"learning_rate":0.01}}})";
    ASSERT_EQ(model.Serialize().dump(), expected);
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
