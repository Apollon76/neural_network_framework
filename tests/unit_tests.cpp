#include <gmock/gmock.h>
#include <src/tensor.hpp>
#include <src/layers/activations.hpp>
#include <src/layers/dense.hpp>
#include <src/layers/convolution2d.hpp>
#include <src/neural_network.hpp>
#include <src/optimizer.hpp>
#include "utils.h"
#include <iostream>

double sigmoidGradient(double x) {
    return exp(-x) / pow((exp(-x) + 1), 2);
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
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
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
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, expected_layer_gradients);
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
    TENSOR_SHOULD_BE_EQUAL_TO(layer.GetWeightsAndBias(), expected);

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
    TENSOR_SHOULD_BE_EQUAL_TO(actual, expected);
}

TEST(DenseLayerTest, TestSerialization) {
    auto layer = DenseLayer<double>(5, 5);
    std::stringstream weights;
    layer.SaveWeights(&weights);
    auto anotherLayer = DenseLayer<double>(5, 5);
    anotherLayer.LoadWeights(&weights);
    TENSOR_SHOULD_BE_EQUAL_TO(layer.GetWeightsAndBias(), anotherLayer.GetWeightsAndBias());
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
    TENSOR_SHOULD_BE_EQUAL_TO(
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
    TENSOR_SHOULD_BE_EQUAL_TO(
            dynamic_cast<DenseLayer<double> *>(network.GetLayer(0))->GetWeightsAndBias(),
            Tensor<double>::init(
                    {
                            {2},
                            {3}
                    }),
            1e-1);
}

TEST(NeuralNetworkTest, TestConvolutionDependency) {
    auto network = NeuralNetwork<double>(
            std::make_unique<Optimizer<double>>(0.01), std::make_unique<MSELoss<double>>()
    );
    network.AddLayer(std::make_unique<Convolution2dLayer<double>>(3, 2, 2, 2, ConvolutionPadding::Same));
    auto expected_layer = Convolution2dLayer<double>(3, 2, 2, 2, ConvolutionPadding::Same);
    auto inputs = Tensor<double>::filled({10, 3, 5, 5}, arma::fill::randu);
    auto outputs = expected_layer.Apply(inputs);
    for (int i = 0; i < 1000; i++) {
        auto loss = network.Fit(inputs, outputs);
        DLOG(INFO) << "Loss: " << loss;
        DLOG(INFO) << "Weights: " << (dynamic_cast<Convolution2dLayer<double> *>(network.GetLayer(0))->GetWeights().ToString());
    }
    TENSOR_SHOULD_BE_EQUAL_TO(
            dynamic_cast<Convolution2dLayer<double> *>(network.GetLayer(0))->GetWeights(),
            expected_layer.GetWeights(),
            1e-1
    );
}

TEST(MSETest, TestLoss) {
    auto loss = MSELoss<double>();
    ASSERT_DOUBLE_EQ(loss.GetLoss(Tensor<double>::init({{1, 2, 3}}), Tensor<double>::init({{5, 9, -1}})),
                     pow(1 - 5, 2) + pow(2 - 9, 2) + pow(3 + 1, 2));
}

TEST(MSETest, TestDerivative) {
    auto loss = MSELoss<double>();
    auto gradients = loss.GetGradients(Tensor<double>::init({{1, 2, 3}}), Tensor<double>::init({{5, 9, -1}}));
    TENSOR_SHOULD_BE_EQUAL_TO(gradients, Tensor<double>::init({{2 * (1 - 5), 2 * (2 - 9), 2 * (3 + 1)}}));
}

TEST(ReLUActivationLayerTest, TestReLULayer) {
    auto layer = ReLUActivationLayer<double>();
    auto input_batch = Tensor<double>::init(
            {
                    {1, 2, 3},
                    {4, 5, -6}
            });
    auto actual = layer.Apply(input_batch);
    TENSOR_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), Tensor<double>::init(
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
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients);
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
    TENSOR_SHOULD_BE_EQUAL_TO(layer.Apply(input_batch), expected,
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
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, Tensor<double>());
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_gradients, 1e-6);
}

TEST(MomentumOptimizerTest, TestDifferentLayers) {
    MomentumOptimizer<double> optimizer(0.5, 0.1);
    auto firstLayer = DenseLayer<double>(1, 1);
    auto secondLayer = DenseLayer<double>(1, 1);

    auto firstLayerGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &firstLayer);
    auto secondLayerGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &secondLayer);

    TENSOR_SHOULD_BE_EQUAL_TO(firstLayerGradientStep, Tensor<double>::init({-0.5, -1, -1.5}));
    TENSOR_SHOULD_BE_EQUAL_TO(secondLayerGradientStep, Tensor<double>::init({-5, -10, -15}));
}

TEST(MomentumOptimizerTest, TestMomentum) {
    MomentumOptimizer<double> optimizer(0.5, 0.1);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({100, 200, 300}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);

    TENSOR_SHOULD_BE_EQUAL_TO(firstGradientStep, Tensor<double>::init({-50, -100, -150}));
    TENSOR_SHOULD_BE_EQUAL_TO(secondGradientStep, Tensor<double>::init({-10, -20, -30}));
    TENSOR_SHOULD_BE_EQUAL_TO(thirdGradientStep, Tensor<double>::init({-6, -12, -18}));
}

TEST(RMSPropOptimizerTest, TestRMSPropGradientStep) {
    RMSPropOptimizer<double> optimizer(0.5, 0.1);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10, 20, 30}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10000, 2, 3}), &layer);

    TENSOR_SHOULD_BE_EQUAL_TO(firstGradientStep, Tensor<double>::init({-0.5270463, -0.5270463, -0.5270463}),
                              1e-6);
    TENSOR_SHOULD_BE_EQUAL_TO(secondGradientStep, Tensor<double>::init({-0.526783, -0.526782, -0.526782}),
                              1e-6);
    TENSOR_SHOULD_BE_EQUAL_TO(thirdGradientStep, Tensor<double>::init({-0.5270462, -0.1588382, -0.158838}),
                              1e-6);
}

TEST(AdamOptimizerTest, TestAdamGradientStep) {
    AdamOptimizer<double> optimizer(0.5);
    auto layer = DenseLayer<double>(1, 1);

    auto firstGradientStep = optimizer.GetGradientStep(Tensor<double>::init({1, 2, 3}), &layer);
    auto secondGradientStep = optimizer.GetGradientStep(Tensor<double>::init({-1, -2, -3}), &layer);
    auto thirdGradientStep = optimizer.GetGradientStep(Tensor<double>::init({10000, 2, 3}), &layer);

    TENSOR_SHOULD_BE_EQUAL_TO(firstGradientStep,
                              Tensor<double>::init({-15.81059779, -15.81119066, -15.81130046}), 1e-6);
    TENSOR_SHOULD_BE_EQUAL_TO(secondGradientStep,
                              Tensor<double>::init({11.18285631, 11.18306609, 11.18310494}), 1e-6);
    TENSOR_SHOULD_BE_EQUAL_TO(thirdGradientStep,
                              Tensor<double>::init({-15.81138814, -9.133237456, -9.133258618}), 1e-6);
}

TEST(Convolution2dLayerTest, TestApply) {
    // 3 входных канала, 2 фильтра
    auto layer = Convolution2dLayer<double>(3, 2, 3, 2, ConvolutionPadding::Same);
    // 2 батча, 3 входных канала
    auto input = Tensor<double>::filled({2, 3, 5, 5}, arma::fill::randu);
    auto output = layer.Apply(input);
    auto expected = Tensor<double>::filled({2, 2, 5, 5}, arma::fill::zeros);
    for (int batch = 0; batch < 2; batch++) {
        for (int filter = 0; filter < 2; filter++) {
            for (int x = 0; x < 5; x++) {
                for (int y = 0; y < 5; y++) {
                    double result = 0;
                    for (int input_channel = 0; input_channel < 3; input_channel++) {
                        for (int dx = 0; dx < 3; dx++) {
                            for (int dy = 0; dy < 2; dy++) {
                                if (x + dx < 5 && y + dy < 5) {
                                    result += input.Field()(batch, input_channel)(x + dx, y + dy) *
                                              layer.GetWeights().Field()(filter, input_channel)(dx, dy);
                                }
                            }
                        }
                    }
                    expected.Field()(batch, filter)(x, y) = result / 3;
                }
            }
        }
    }
    TENSOR_SHOULD_BE_EQUAL_TO(output, expected);
}

TEST(Convolution2dLayerTest, TestApplyGradient) {
    auto layer = Convolution2dLayer<double>(2, 2, 2, 2, ConvolutionPadding::Same);
    auto gradients = Tensor<double>::init(
            {
                    {
                            {
                                    {1,  2},
                                    {3,  4}
                            },
                            {
                                    {5,  6},
                                    {7,  8}
                            }
                    },
                    {
                            {
                                    {-1, -2},
                                    {-3, -4}
                            },
                            {
                                    {-5, -6},
                                    {-7, -8}
                            }
                    }
            });
    arma::field<arma::Mat<double>> weights = layer.GetWeights().Field();
    layer.ApplyGradients(gradients);
    TENSOR_SHOULD_BE_EQUAL_TO(layer.GetWeights(), Tensor<double>::init(
            {
                    {
                            {
                                    {weights(0, 0)(0, 0) + 1, weights(0, 0)(0, 1) + 2},
                                    {weights(0, 0)(1, 0) + 3, weights(0, 0)(1, 1) + 4}
                            },
                            {
                                    {weights(0, 1)(0, 0) + 5, weights(0, 1)(0, 1) + 6},
                                    {weights(0, 1)(1, 0) + 7, weights(0, 1)(1, 1) + 8}
                            },
                    },
                    {
                            {
                                    {weights(1, 0)(0, 0) - 1, weights(1, 0)(0, 1) - 2},
                                    {weights(1, 0)(1, 0) - 3, weights(1, 0)(1, 1) - 4}
                            },
                            {
                                    {weights(1, 1)(0, 0) - 5, weights(1, 1)(0, 1) - 6},
                                    {weights(1, 1)(1, 0) - 7, weights(1, 1)(1, 1) - 8}
                            },
                    }
            }
    ));
}

TEST(Convolution2dLayerTest, TestPullGradientsBackward) {
    auto layer = Convolution2dLayer<double>(3, 2, 2, 2, ConvolutionPadding::Same);
    auto input = Tensor<double>::filled({4, 3, 5, 5}, arma::fill::randu);
    auto output_gradients = Tensor<double>::filled({4, 2, 5, 5}, arma::fill::randu);
    auto gradients = layer.PullGradientsBackward(input, output_gradients);
    auto expected_layer_grad = Tensor<double>::filled({2, 3, 2, 2}, arma::fill::zeros);
    auto expected_input_grad = Tensor<double>::filled({4, 3, 5, 5}, arma::fill::zeros);
    for (int batch = 0; batch < 4; batch++) {
        for (int input_channel = 0; input_channel < 3; input_channel++) {
            for (int filter = 0; filter < 2; filter++) {
                for (int x = 0; x < 5; x++) {
                    for (int y = 0; y < 5; y++) {
                        for (int dx = 0; dx < 2; dx++) {
                            for (int dy = 0; dy < 2; dy++) {
                                if (x + dx >= 5 || y + dy >= 5) {
                                    continue;
                                }
                                // output[batch, filter, x + dx, y + dy] += input[batch, input_channel, x, y] * w[filter, input_channel, dx, dy] / input_channels
                                expected_layer_grad.Field()(filter, input_channel)(dx, dy) +=
                                        output_gradients.Field()(batch, filter)(x + dx, y + dy) *
                                        input.Field()(batch, input_channel)(x, y) / 3;
                                expected_input_grad.Field()(batch, input_channel)(x, y) +=
                                        output_gradients.Field()(batch, filter)(x + dx, y + dy) *
                                        layer.GetWeights().Field()(filter, input_channel)(dx, dy) / 3;
                            }
                        }
                    }
                }
            }
        }
    }
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.input_gradients, expected_input_grad);
    TENSOR_SHOULD_BE_EQUAL_TO(gradients.layer_gradients, expected_layer_grad);
}

TEST(Convolution2dLayerTest, TestSerialization) {

}

TEST(ArmaConvolutionTest, TestValid) {
    auto image = arma::mat(3, 3, arma::fill::randu);
    auto kernel = arma::mat(1, 2, arma::fill::randu);
    auto result = Conv2d(image, kernel, ConvolutionPadding::Valid);
    MATRIX_SHOULD_BE_EQUAL_TO(result, arma::mat
            ({
                     {
                             image(0, 0) * kernel(0, 0) + image(0, 1) * kernel(0, 1),
                             image(0, 1) * kernel(0, 0) + image(0, 2) * kernel(0, 1),
                     },
                     {
                             image(1, 0) * kernel(0, 0) + image(1, 1) * kernel(0, 1),
                             image(1, 1) * kernel(0, 0) + image(1, 2) * kernel(0, 1),
                     },
                     {
                             image(2, 0) * kernel(0, 0) + image(2, 1) * kernel(0, 1),
                             image(2, 1) * kernel(0, 0) + image(2, 2) * kernel(0, 1),
                     },
             })
    );
}

TEST(ArmaConvolutionTest, TestSame) {
    auto image = arma::mat(3, 3, arma::fill::randu);
    auto kernel = arma::mat(1, 2, arma::fill::randu);
    auto result = Conv2d(image, kernel, ConvolutionPadding::Same);
    MATRIX_SHOULD_BE_EQUAL_TO(result, arma::mat
            ({
                     {
                             image(0, 0) * kernel(0, 0) + image(0, 1) * kernel(0, 1),
                             image(0, 1) * kernel(0, 0) + image(0, 2) * kernel(0, 1),
                             image(0, 2) * kernel(0, 0),
                     },
                     {
                             image(1, 0) * kernel(0, 0) + image(1, 1) * kernel(0, 1),
                             image(1, 1) * kernel(0, 0) + image(1, 2) * kernel(0, 1),
                             image(1, 2) * kernel(0, 0),
                     },
                     {
                             image(2, 0) * kernel(0, 0) + image(2, 1) * kernel(0, 1),
                             image(2, 1) * kernel(0, 0) + image(2, 2) * kernel(0, 1),
                             image(2, 2) * kernel(0, 0),
                     },
             })
    );
}