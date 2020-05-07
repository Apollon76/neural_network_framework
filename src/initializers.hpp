#pragma once
#include <random>
#include "tensor.hpp"

namespace {
    std::random_device rd;
    std::mt19937 generator{rd()};

    class truncated_normal_distribution {
    public:
        truncated_normal_distribution(double mean, double stddev)
                : distribution(std::normal_distribution(mean, stddev)), stddev(stddev)
        {
        }

        template <class T>
        double operator()(T& generator) {
            while (true) {
                double result = distribution(generator);
                if (fabs(result) < 2 * stddev) {
                    return result;
                }
            }
        }

    private:
        std::normal_distribution<double> distribution;
        double stddev;
    };
}

class IInitializer {
public:
    virtual Tensor<double> generate(TensorDimensions dimensions) = 0;
};

class ConstInitializer : public  IInitializer {
public:
    explicit ConstInitializer(double value)
            : value(value)
    {
    }

    Tensor<double> generate(TensorDimensions dimensions) override {
        return Tensor<double>::filled(dimensions, value);
    }
private:
    double value;
};

class UniformInitializer : public IInitializer {
public:
    explicit UniformInitializer(double minval = -0.5, double maxval = 0.5)
            : minval(minval), maxval(maxval)
    {
    }

    Tensor<double> generate(TensorDimensions dimensions) override {
        std::uniform_real_distribution<double> distribution(minval, maxval);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
private:
    double minval, maxval;
};

class NormalInitializer : public IInitializer {
public:
    explicit NormalInitializer(double mean = 0, double stddev = 0.05)
            : mean(mean), stddev(stddev)
    {
    }

    Tensor<double> generate(TensorDimensions dimensions) override {
        truncated_normal_distribution distribution(mean, stddev);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
private:
    double mean, stddev;
};

class GlorotNormalInitializer : public IInitializer {
public:
    Tensor<double> generate(TensorDimensions dimensions) override {
        truncated_normal_distribution distribution(0, sqrt(2.0 / (dimensions.front() + dimensions.back())));
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
};

class GlorotUniformInitializer : public IInitializer {
public:
    Tensor<double> generate(TensorDimensions dimensions) override {
        double limit = sqrt(6.0 / (dimensions.front() + dimensions.back()));
        std::uniform_real_distribution<double> distribution(-limit, limit);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
};
