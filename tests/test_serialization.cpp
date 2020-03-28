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

        auto optimizer = AdamOptimizer(0.01, 0.8, 0.7, 1e-5);
        oarchive(optimizer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        AdamOptimizer deserialized;
        iarchive(deserialized);
    }
}

TEST(SerializationTest, TestDenseAsLayer) {
    auto filename = "layer_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        std::shared_ptr<ILayer> layer = std::make_shared<DenseLayer>(4, 5);
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        std::shared_ptr<ILayer> deserialized;
        iarchive(deserialized);
    }
}

TEST(SerializationTest, TestReLUASLayer) {
    auto filename = "layer_test.json";
    {
        std::ofstream os(filename);
        cereal::JSONOutputArchive oarchive(os);

        std::shared_ptr<ILayer> layer = std::make_shared<ReLUActivationLayer>();
        oarchive(layer);
    }

    {
        std::ifstream is(filename);
        cereal::JSONInputArchive iarchive(is);

        std::shared_ptr<ILayer> deserialized;
        iarchive(deserialized);

        auto input_batch = arma::mat(
                {
                        {1, 2, 3},
                        {4, 5, -6}
                });
        arma::mat actual = deserialized->Apply(input_batch);
        MATRIX_SHOULD_BE_EQUAL_TO(actual, arma::mat(
                {
                        {1, 2, 3},
                        {4, 5, 0}
                }));
    }
}

TEST(SerializationTest, TestSequentialModel) {
    auto model = NeuralNetwork(std::make_unique<Optimizer>(0.01), std::make_unique<MSELoss>());;
    model.AddLayer(std::make_unique<DenseLayer>(2, 3));
    model.AddLayer(std::make_unique<SigmoidActivationLayer>());
    model.AddLayer(std::make_unique<DenseLayer>(3, 1));
    model.AddLayer(std::make_unique<ReLUActivationLayer>());

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

        NeuralNetwork deserialized;
        iarchive(deserialized);

        std::vector<std::string> actual_names;
        for (size_t i = 0; i < 4; ++i) {
            actual_names.push_back(deserialized.GetLayer(i)->GetName());
        }
        ASSERT_EQ(actual_names, expected_names);
    }
}