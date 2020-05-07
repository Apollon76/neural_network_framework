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
        double operator()(T& gen) {
            while (true) {
                double result = distribution(gen);
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
    [[nodiscard]] virtual Tensor<double> generate(TensorDimensions dimensions) const = 0;
    virtual ~IInitializer() = default;
};

class ConstInitializer : public  IInitializer {
public:
    explicit ConstInitializer(double value)
            : value(value)
    {
    }

    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        int val = value;
        return Tensor<double>::filledRandom(dimensions, [val](){return val; });
    }

    ~ConstInitializer() override = default;
private:
    double value;
};

class UniformInitializer : public IInitializer {
public:
    explicit UniformInitializer(double minval = -0.5, double maxval = 0.5)
            : minval(minval), maxval(maxval)
    {
    }

    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        std::uniform_real_distribution<double> distribution(minval, maxval);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }

    ~UniformInitializer() override = default;
private:
    double minval, maxval;
};

class NormalInitializer : public IInitializer {
public:
    explicit NormalInitializer(double mean = 0, double stddev = 0.05)
            : mean(mean), stddev(stddev)
    {
    }

    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        truncated_normal_distribution distribution(mean, stddev);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }

    ~NormalInitializer() override = default;
private:
    double mean, stddev;
};

class GlorotNormalInitializer : public IInitializer {
public:
    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        truncated_normal_distribution distribution(0, sqrt(2.0 / (dimensions.front() + dimensions.back())));
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
};

class GlorotUniformInitializer : public IInitializer {
public:
    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        double limit = sqrt(6.0 / (dimensions.front() + dimensions.back()));
        std::uniform_real_distribution<double> distribution(-limit, limit);
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
};
