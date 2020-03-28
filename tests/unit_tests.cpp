#include <gmock/gmock.h>
#include <src/tensor.hpp>
#include <src/layers/activations.hpp>
#include <src/layers/dense.hpp>
#include <src/neural_network.hpp>
#include <src/optimizer.hpp>

double sigmoidGradient(double x) {
    return exp(-x) / pow((exp(-x) + 1), 2);
}

const double eps = 1e-9;

template<typename T>
void MATRIX_SHOULD_BE_EQUAL_TO(const Tensor<T> &actual, const Tensor<T> &expected, double tolerance = eps) {
    std::stringstream message;
    message << "Expected matrix: " << std::endl << expected << "Bug given matrix: " << std::endl << actual;
    ASSERT_TRUE(arma::approx_equal(actual.Values(), expected.Values(), "both", tolerance, tolerance)) << message.str();
}

TEST(SigmoidActivationLayerTest, TestPullGradientsBackward) {
    auto layer = SigmoidActivationLayer<double>();
    auto input_batch = Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            });
    auto output_gradients = Tensor<double>::init(
            {
                    {10, 20, 30},
                    {40, 50, 60}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = Tensor<double>::init(
            {
                    {10 * sigmoidGradient(1), 20 * sigmoidGradient(2), 30 * sigmoidGradient(3)},
                    {40 * sigmoidGradient(4), 50 * sigmoidGradient(5), 60 * sigmoidGradient(6)},
            }
    );
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
}


TEST(DenseLayerTest, TestPullGradientsBackward) {
    auto layer = DenseLayer<double>(3, 2);
    auto i = Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            }
    );
    auto og = Tensor<double>::init(
            {
                    {7, 8},
                    {9, 10}
            }
    );
    auto gradients = layer.PullGradientsBackward(i, og);
    auto expected_layer_gradients = Tensor<double>::init(
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
    auto layer = DenseLayer<double>(2, 2);
    auto w = layer.GetWeightsAndBias();
    auto g = Tensor<double>::init(
            {
                    {1, 2},
                    {3, 4},
                    {5, 6}
            }
    );
    auto expected = Tensor<double>::init(
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
    auto layer = DenseLayer<double>(3, 2);
    auto w = layer.GetWeightsAndBias();
    auto i = Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, 6}
            }
    );
    auto expected = Tensor<double>::init(
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

TEST(DenseLayerTest, TestSerialization) {
    auto layer = DenseLayer<double>(5, 5);
    std::stringstream weights;
    layer.SaveWeights(&weights);
    auto anotherLayer = DenseLayer<double>(5, 5);
    anotherLayer.LoadWeights(&weights);
    MATRIX_SHOULD_BE_EQUAL_TO(layer.GetWeightsAndBias(), anotherLayer.GetWeightsAndBias());
}

TEST(NeuralNetworkTest, TestLinearDependency) {
    auto network = NeuralNetwork<double>(
            std::make_unique<Optimizer<double>>(0.01), std::make_unique<MSELoss<double>>()
    );
    network.AddLayer(std::make_unique<DenseLayer<double>>(1, 1));
    auto inputs = Tensor<double>::init(
            {
                    {1},
                    {5}
            });
    auto outputs = Tensor<double>::init(
            {
                    {2 * 1 + 3},
                    {2 * 5 + 3}
            });
    for (int i = 0; i < 10000; i++) {
        network.Fit(inputs, outputs);
    }
    MATRIX_SHOULD_BE_EQUAL_TO(
            dynamic_cast<DenseLayer<double> *>(network.GetLayer(0))->GetWeightsAndBias(),
            Tensor<double>::init(
                    {
                            {2},
                            {3}
                    }),
            1e-5);
}

TEST(NeuralNetworkTest, TestLinearDependencyWithSigmoid) {
    auto network = NeuralNetwork<double>(
            std::make_unique<Optimizer<double>>(0.1), std::make_unique<MSELoss<double>>()
    );
    network.AddLayer(std::make_unique<DenseLayer<double>>(1, 1));
    network.AddLayer(std::make_unique<SigmoidActivationLayer<double>>());
    auto inputs = Tensor<double>::init(
            {
                    {-2},
                    {-1},
                    {0},
                    {1},
                    {2},
            });
    auto outputs = Tensor<double>::init(
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
            dynamic_cast<DenseLayer<double> *>(network.GetLayer(0))->GetWeightsAndBias(),
            Tensor<double>::init(
                    {
                            {2},
                            {3}
                    }),
            1e-1);
}

TEST(MSETest, TestLoss) {
    auto loss = MSELoss<double>();
    ASSERT_DOUBLE_EQ(loss.GetLoss(Tensor<double>::init({{1, 2, 3}}), Tensor<double>::init({{5, 9, -1}})),
                     pow(1 - 5, 2) + pow(2 - 9, 2) + pow(3 + 1, 2));
}

TEST(MSETest, TestDerivative) {
    auto loss = MSELoss<double>();
    auto gradients = loss.GetGradients(Tensor<double>::init({{1, 2, 3}}), Tensor<double>::init({{5, 9, -1}}));
    MATRIX_SHOULD_BE_EQUAL_TO(gradients, Tensor<double>::init({{2 * (1 - 5), 2 * (2 - 9), 2 * (3 + 1)}}));
}

TEST(ReLUActivationLayerTest, TestReLULayer) {
    auto layer = ReLUActivationLayer<double>();
    auto input_batch = Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, -6}
            });
    auto actual = layer.Apply(input_batch);
    MATRIX_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, 0}
            }));

    auto output_gradients = Tensor<double>::init(
            {
                    {10, 20, 30},
                    {40, 50, 60}
            });
    auto gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = Tensor<double>::init(
            {
                    {10, 20, 30},
                    {40, 50, 0}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
}

TEST(TanhActivationLayerTest, TestTanhLayer) {
    TanhActivationLayer layer = TanhActivationLayer<double>();
    auto input_batch = Tensor<double>::init(
            {
                    {1,  2,  3},
                    {-1, -2, -3}
            });
    auto actual = layer.Apply(input_batch);
    auto expected = Tensor<double>::init(
            {
                    {0.7615942,  0.9640276,  0.9950548},
                    {-0.7615942, -0.9640276, -0.9950548}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), expected,
                              1e-6
    );

    auto output_gradients = Tensor<double>::init(
            {
                    {10,  20,  30},
                    {-10, -20, -30}
            });
    Gradients gradients = layer.PullGradientsBackward(input_batch, output_gradients);
    auto expected_gradients = Tensor<double>::init(
            {
                    {4.199743,  1.4130175,  0.29598176},
                    {-4.199743, -1.4130175, -0.29598176}
            });
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    MATRIX_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients, 1e-6);
}

TEST(MomentumOptimizerTest, TestDifferentLayers) {
    MomentumOptimizer<double> optimizer(0.5, 0.1);
    auto firstLayer = DenseLayer<double>(1, 1);
    auto secondLayer = DenseLayer<double>(1, 1);

    auto firstLayerGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &firstLayer);
    auto secondLayerGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &secondLayer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstLayerGradientStep, Tensor<double>::init({-0.5, -1, -1.5}));
    MATRIX_SHOULD_BE_EQUAL_TO(secondLayerGradientStep, Tensor<double>::init({-5, -10, -15}));
}

TEST(MomentumOptimizerTest, TestMomentum) {
    MomentumOptimizer<double> optimizer(0.5, 0.1);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({100, 200, 300}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep, Tensor<double>::init({-50, -100, -150}));
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep, Tensor<double>::init({-10, -20, -30}));
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep, Tensor<double>::init({-6, -12, -18}));
}

TEST(RMSPropOptimizerTest, TestRMSPropGradientStep) {
    RMSPropOptimizer<double> optimizer(0.5, 0.1);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10000, 2, 3}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep, Tensor<double>::init({-0.5270463, -0.5270463, -0.5270463}),
                              1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep, Tensor<double>::init({-0.526783, -0.526782, -0.526782}),
                              1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep, Tensor<double>::init({-0.5270462, -0.1588382, -0.158838}),
                              1e-6);
}

TEST(AdamOptimizerTest, TestAdamGradientStep) {
    AdamOptimizer<double> optimizer(0.5);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({-1, -2, -3}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10000, 2, 3}), &layer);

    MATRIX_SHOULD_BE_EQUAL_TO(firstGradientStep,
                              Tensor<double>::init({-15.81059779, -15.81119066, -15.81130046}), 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(secondGradientStep,
                              Tensor<double>::init({11.18285631, 11.18306609, 11.18310494}), 1e-6);
    MATRIX_SHOULD_BE_EQUAL_TO(thirdGradientStep,
                              Tensor<double>::init({-15.81138814, -9.133237456, -9.133258618}), 1e-6);
}

TEST(SerializationTest, TestNNSerialization) {
    auto model = NeuralNetwork<double>(
            std::make_unique<Optimizer<double>>(0.01), std::make_unique<MSELoss<double>>()
    );;
    model.AddLayer(std::make_unique<DenseLayer<double>>(2, 3));
    model.AddLayer(std::make_unique<SigmoidActivationLayer<double>>());
    model.AddLayer(std::make_unique<DenseLayer<double>>(3, 1));
    model.AddLayer(std::make_unique<ReLUActivationLayer<double>>());

    auto expected = R"({"layers":[{"layer_type":"dense","params":{"n_cols":3,"n_rows":2}},{"layer_type":"sigmoid_activation"},{"layer_type":"dense","params":{"n_cols":1,"n_rows":3}},{"layer_type":"relu_activation"}],"loss":["loss_type","mse"],"optimizer":{"optimizer_type":"optimizer","params":{"learning_rate":0.01}}})";
    ASSERT_EQ(model.Serialize().dump(), expected);
}
