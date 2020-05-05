#include <fstream>
#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>
#include <src/optimizer.hpp>
#include <src/layers/dense.hpp>
#include <src/loss.hpp>
#include <src/neural_network.hpp>
#include <src/layers/flatten.hpp>
#include <src/layers/convolution2d.hpp>
#include <src/arma_math.hpp>

#include "utils.h"


TEST(SerializationTest, TestSaveOptimizer) {
    auto filename = "optimizer_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        auto optimizer = AdamOptimizer<double>(0.01, 0.8, 0.7, 1e-5);
        oarchive(optimizer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        AdamOptimizer<double> deserialized;
        iarchive(deserialized);
    }
}

TEST(SerializationTest, TestDenseAsLayer) {
    auto filename = "layer_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        std::shared_ptr<ILayer<double>> layer = std::make_shared<DenseLayer<double>>(4, 5);
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        std::shared_ptr<ILayer<double>> deserialized;
        iarchive(deserialized);
    }
}

TEST(SerializationTest, TestReLUASLayer) {
    auto filename = "layer_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        std::shared_ptr<ILayer<double>> layer = std::make_shared<ReLUActivationLayer<double>>();
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        std::shared_ptr<ILayer<double>> deserialized;
        iarchive(deserialized);

        auto input_batch = Tensor<double>::init(
                {
                        {1, 2, 3},
                        {4, 5, -6}
                });
        auto actual = deserialized->Apply(input_batch);
        TENSOR_SHOULD_BE_EQUAL_TO(actual, Tensor<double>::init(
                {
                        {1, 2, 3},
                        {4, 5, 0}
                }));
    }
}

TEST(SerializationTest, TestSequentialModel) {
    auto model = NeuralNetwork<double>(std::make_unique<Optimizer<double>>(0.01), std::make_unique<MSELoss<double>>());;
    model.AddLayer(std::make_unique<DenseLayer<double>>(2, 3));
    model.AddLayer(std::make_unique<SigmoidActivationLayer<double>>());
    model.AddLayer(std::make_unique<DenseLayer<double>>(3, 1));
    model.AddLayer(std::make_unique<ReLUActivationLayer<double>>());

    std::vector<std::string> expected_names;
    for (size_t i = 0; i < 4; ++i) {
        expected_names.push_back(model.GetLayer(i)->GetName());
    }

    auto filename = "model_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        oarchive(model);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        NeuralNetwork<double> deserialized;
        iarchive(deserialized);

        std::vector<std::string> actual_names;
        for (size_t i = 0; i < 4; ++i) {
            actual_names.push_back(deserialized.GetLayer(i)->GetName());
        }
        ASSERT_EQ(actual_names, expected_names);
    }
}

TEST(SerializationTest, TestSaveFlatten) {
    auto filename = "flatten_test.json";
    TensorDimensions shape = {1, 2, 3};
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        auto layer = FlattenLayer<double>(shape);
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        FlattenLayer<double> deserialized;
        iarchive(deserialized);

        ASSERT_EQ(deserialized.ToString(), "Flatten, input dimensions: 1 x 2 x 3");
    }
}

TEST(SerializationTest, TestSaveConv2d) {
    auto filename = "conv2d_test.json";
    TensorDimensions shape = {1, 2, 3};
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        auto layer = Convolution2dLayer<double>(1, 2, 3, 4, ConvolutionPadding::Valid);
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        Convolution2dLayer<double> deserialized;
        iarchive(deserialized);

        ASSERT_EQ(deserialized.GetName(), "Conv2d[2 x 1 x 3 x 4 (including bias)]");
    }
}
