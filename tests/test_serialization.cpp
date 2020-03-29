#include <fstream>
#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>
#include <src/optimizer.hpp>
#include <src/layers/dense.hpp>
#include <src/loss.hpp>
#include <src/neural_network.hpp>
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
        MATRIX_SHOULD_BE_EQUAL_TO(actual, Tensor<double>::init(
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
    std::vector<std::string> expected_idx;
    for (size_t i = 0; i < 4; ++i) {
        expected_names.push_back(model.GetLayer(i)->GetName());
        expected_idx.push_back(model.GetLayer(i)->GetLayerID());
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
        std::vector<std::string> actual_idx;
        for (size_t i = 0; i < 4; ++i) {
            actual_names.push_back(deserialized.GetLayer(i)->GetName());
            actual_idx.push_back(deserialized.GetLayer(i)->GetLayerID());
        }
        ASSERT_EQ(actual_names, expected_names);
        ASSERT_EQ(actual_idx, expected_idx);
    }
}