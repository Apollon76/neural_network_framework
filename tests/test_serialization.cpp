#include <fstream>
#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>
#include <src/optimizer.hpp>


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
