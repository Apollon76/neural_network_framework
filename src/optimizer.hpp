#pragma once

#include <armadillo>
#include <cereal/types/polymorphic.hpp>
#include "utils.hpp"
#include "layers/interface.h"


class IOptimizer {
public:
    [[nodiscard]] virtual arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) = 0;

    virtual ~IOptimizer() = default;
};

class Optimizer : public IOptimizer {
public:
    Optimizer() {}
    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) override {
        UNUSED(layer)
        return -gradients * learning_rate;
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar(learning_rate);
    }

private:
    double learning_rate;
};

CEREAL_REGISTER_TYPE(Optimizer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer, Optimizer)

class MomentumOptimizer : public IOptimizer {
public:
    MomentumOptimizer() {}
    explicit MomentumOptimizer(double _learning_rate, double _momentum)
            : learning_rate(_learning_rate), momentum(_momentum) {
    }

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) override {
        auto it = previous_values.find(layer);
        if (it == previous_values.end()) {
            return previous_values[layer] = -learning_rate * gradients;
        }
        auto previous_gradient = it->second;
        return previous_values[layer] = momentum * previous_gradient - learning_rate * gradients;
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar(learning_rate, momentum);
    }

private:
    double learning_rate;
    double momentum;

    std::unordered_map<const ILayer*, arma::mat> previous_values;
};

CEREAL_REGISTER_TYPE(MomentumOptimizer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer, MomentumOptimizer)

class RMSPropOptimizer : public IOptimizer {
public:
    explicit RMSPropOptimizer(double _learning_rate = 0.001, double _rho = 0.9, double _eps = 1e-7)
            : learning_rate(_learning_rate), rho(_rho), epsilon(_eps) {
    }

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat& gradients, const ILayer* layer) override {
        auto it = previous_mean.find(layer);
        arma::mat previous_gradient;
        if (it == previous_mean.end()) {
            previous_gradient.zeros(arma::size(gradients));
        } else {
            previous_gradient = it->second;
        }

        auto currentMean = previous_mean[layer] = rho * previous_gradient + (1 - rho) * arma::square(gradients);
        return -learning_rate * (gradients / arma::sqrt(currentMean + epsilon));
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar(learning_rate, rho, epsilon);
    }

private:
    double learning_rate;
    double rho;
    double epsilon;

    std::unordered_map<const ILayer*, arma::mat> previous_mean;
};

CEREAL_REGISTER_TYPE(RMSPropOptimizer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer, RMSPropOptimizer)

class AdamOptimizer : public IOptimizer {
    using mapping_type = std::unordered_map<const ILayer *, arma::mat>;
public:
    explicit AdamOptimizer(double _learning_rate = 0.001, double _beta_1 = 0.9, double _beta_2 = 0.999,
                           double _eps = 1e-7)
            : learning_rate(_learning_rate), beta_1(_beta_1), beta_2(_beta_2), epsilon(_eps) {
    }

    [[nodiscard]] arma::mat GetGradientStep(const arma::mat &gradients, const ILayer *layer) override {
        auto previous_gradient_avg = GetOrCreatePrevious(layer, average_of_gradients, arma::size(gradients));
        auto previous_square_gradient_avg = GetOrCreatePrevious(layer, average_of_squares_of_gradients,
                                                                arma::size(gradients));

        auto cur_avg = average_of_gradients[layer] = beta_1 * previous_gradient_avg + (1 - beta_1) * gradients;
        auto cur_square_avg = average_of_squares_of_gradients[layer] =
                                      beta_2 * previous_square_gradient_avg + (1 - beta_2) * arma::square(gradients);

        return -learning_rate * gradients / arma::sqrt(cur_square_avg + epsilon);
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar(learning_rate, beta_1, beta_2, epsilon);
    }

private:
    arma::mat GetOrCreatePrevious(const ILayer *layer, mapping_type &mapping, arma::SizeMat size) {
        auto it = mapping.find(layer);
        arma::mat previous;
        if (it != mapping.end()) {
            return it->second;
        }
        previous.zeros(size);
        return previous;
    }

    double learning_rate;
    double beta_1;
    double beta_2;
    double epsilon;
    mapping_type average_of_gradients;
    mapping_type average_of_squares_of_gradients;
};

CEREAL_REGISTER_TYPE(AdamOptimizer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer, AdamOptimizer)
