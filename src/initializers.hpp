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

template <class T>
class IInitializer {
public:
    [[nodiscard]] virtual Tensor<T> generate(TensorDimensions dimensions) const = 0;
    virtual ~IInitializer() = default;
};

class ConstInitializer : public  IInitializer<double> {
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

class UniformInitializer : public IInitializer<double> {
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

class NormalInitializer : public IInitializer<double> {
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

class GlorotNormalInitializer : public IInitializer<double> {
public:
    [[nodiscard]] Tensor<double> generate(TensorDimensions dimensions) const override {
        truncated_normal_distribution distribution(0, sqrt(2.0 / (dimensions.front() + dimensions.back())));
        return Tensor<double>::filledRandom(dimensions, [&distribution](){return distribution(generator); });
    }
};

template<class T>
class GlorotUniformInitializer : public IInitializer<T> {
public:
    [[nodiscard]] Tensor<T> generate(TensorDimensions dimensions) const override {
        ensure(!dimensions.empty());
        double limit;
        if (dimensions.size() == 1) {
            limit = sqrt(3.0 / (dimensions[0]));
        } else if (dimensions.size() == 2) {
            limit = sqrt(6.0 / (dimensions[0] + dimensions[1]));
        } else {
            /*
             * Our conv2d weights have shape: (filters, input_channels, kernel_height, kernel_width)
             * So, receptive_field_size = mult(shape[2:])
             * */
            double receptive_field_size = 1;
            for (size_t i = 2; i < dimensions.size(); i++) {
                receptive_field_size *= dimensions[i];
            }
            limit = sqrt(6.0 / ((dimensions[0] + dimensions[1]) * receptive_field_size));
        }
        std::uniform_real_distribution<T> distribution(-limit, limit);
        return Tensor<T>::filledRandom(dimensions, [&distribution]() { return distribution(generator); });
    }
};
