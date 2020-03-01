#pragma once

#include <armadillo>

class IOptimizer {
public:
    [[nodiscard]] virtual arma::mat GetGradientStep(const arma::mat &gradients) const = 0;
};

class Optimizer : public IOptimizer {
public:
    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat &gradients) const override {
        return -gradients * learning_rate;
    }

private:
    double learning_rate;
};