#pragma once

#include <armadillo>
#include "utils.hpp"

class IOptimizer {
public:
    [[nodiscard]] virtual arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) = 0;

    virtual ~IOptimizer() = default;
};

class Optimizer : public IOptimizer {
public:
    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat &gradients, const ILayer *layer) override {
        UNUSED(layer)
        return -gradients * learning_rate;
    }

private:
    double learning_rate;
};

class MomentumOptimizer : public IOptimizer {
public:
    explicit MomentumOptimizer(double _learning_rate, double _momentum)
            : learning_rate(_learning_rate), momentum(_momentum) {
    }

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) override {
        auto it = previous_values.find(layer);
        if (it == previous_values.end()) {
            auto ret = -learning_rate * gradients;
            previous_values[layer] = ret;
            return ret;
        }
        auto previous_gradient = it->second;

        return momentum * previous_gradient - learning_rate * gradients;
    }

private:
    double learning_rate;
    double momentum;

    std::unordered_map<const ILayer*, arma::mat> previous_values;
};