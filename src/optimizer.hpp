#pragma once

#include <armadillo>

#include <cereal/types/polymorphic.hpp>
#include "utils.hpp"
#include "layers/interface.h"
#include <src/tensor.hpp>

template<typename T>
class IOptimizer {
public:
    [[nodiscard]] virtual Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) = 0;
    [[nodiscard]] virtual Tensor<T> GetGradientStep(Tensor<T>&& gradients, const ILayer<T> *layer) = 0;

    virtual ~IOptimizer() = default;
};

template<typename T>
class Optimizer : public IOptimizer<T> {
public:
    Optimizer() {}

    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        return GetGradientStep(std::move(Tensor<T>(gradients)), layer);
    }

    [[nodiscard]] Tensor<T> GetGradientStep(Tensor<T>&& gradients, const ILayer<T> *) override {
        gradients.ForEach([this](int, int, int, arma::Mat<T> &v) {
            v *= -learning_rate;
        });
        return std::move(gradients);
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(learning_rate);
    }

private:
    double learning_rate;
};

CEREAL_REGISTER_TYPE(Optimizer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer<double>, Optimizer<double>)

template<typename T>
class MomentumOptimizer : public IOptimizer<T> {
public:
    MomentumOptimizer() {}

    explicit MomentumOptimizer(double _learning_rate, double _momentum)
            : learning_rate(_learning_rate), momentum(_momentum) {
    }

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        auto it = previous_values.find(layer);
        if (it == previous_values.end()) {
            return previous_values[layer] = Tensor<T>(gradients.D, -learning_rate * gradients.Values());
        }
        auto previous_gradient = it->second;
        return previous_values[layer] = Tensor<T>(
                gradients.D, momentum * previous_gradient.Values() - learning_rate * gradients.Values()
        );
    }

    [[nodiscard]] Tensor<T> GetGradientStep(Tensor<T>&& gradients, const ILayer<T> *layer) override {
        const Tensor<T> tensor(gradients);
        return GetGradientStep(tensor, layer);
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(learning_rate, momentum);
    }

private:
    double learning_rate;
    double momentum;

    std::unordered_map<const ILayer<T> *, Tensor<T>> previous_values;
};

CEREAL_REGISTER_TYPE(MomentumOptimizer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer<double>, MomentumOptimizer<double>)

template<typename T>
class RMSPropOptimizer : public IOptimizer<T> {
public:
    explicit RMSPropOptimizer(double _learning_rate = 0.001, double _rho = 0.9, double _eps = 1e-7)
            : learning_rate(_learning_rate), rho(_rho), epsilon(_eps) {
    }

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        auto it = previous_mean.find(layer);
        Tensor<T> previous_gradient;
        if (it == previous_mean.end()) {
            previous_gradient = Tensor<T>::filled(gradients.D, arma::fill::zeros);
        } else {
            previous_gradient = it->second;
        }

        auto currentMean = previous_mean[layer] = previous_gradient.template DiffWith<T>(
                gradients, [this](const arma::Mat<T> &a,
                                  const arma::Mat<T> &b) {
                    arma::Mat<T> result = rho * a + (1 - rho) * arma::square(b);
                    return result;
                });
        return gradients.template DiffWith<T>(currentMean, [this](const arma::Mat<T> &a, const arma::Mat<T> &b) {
            arma::Mat<T> result = -learning_rate * (a / arma::sqrt(b + epsilon));
            return result;
        });
    }

    [[nodiscard]] Tensor<T> GetGradientStep(Tensor<T>&& gradients, const ILayer<T> *layer) override {
        const Tensor<T> tensor(gradients);
        return GetGradientStep(tensor, layer);
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(learning_rate, rho, epsilon);
    }

private:
    double learning_rate;
    double rho;
    double epsilon;

    std::unordered_map<const ILayer<T> *, Tensor<T>> previous_mean;
};

CEREAL_REGISTER_TYPE(RMSPropOptimizer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer<double>, RMSPropOptimizer<double>)

template<typename T>
class AdamOptimizer : public IOptimizer<T> {
    using mapping_type = std::unordered_map<const ILayer<T> *, arma::Mat<T>>;
public:
    explicit AdamOptimizer(double _learning_rate = 0.001, double _beta_1 = 0.9, double _beta_2 = 0.999,
                           double _eps = 1e-7)
            : learning_rate(_learning_rate), beta_1(_beta_1), beta_2(_beta_2), epsilon(_eps) {
    }

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        return GetGradientStep(std::move(Tensor<T>(gradients)), layer);
    }

    [[nodiscard]] Tensor<T> GetGradientStep(Tensor<T>&& gradients, const ILayer<T> *layer) override {
        if (!average_of_squares_of_gradients.count(layer)) {
            average_of_squares_of_gradients[layer].zeros(arma::size(gradients.Values()));
        }
        auto& layer_average_of_squares_of_gradients = average_of_squares_of_gradients[layer];

        layer_average_of_squares_of_gradients *= beta_2;
        layer_average_of_squares_of_gradients += (1 - beta_2) * arma::square(gradients.Values());

        gradients.ForEach([this, &layer_average_of_squares_of_gradients](int, int, int, arma::Mat<T> &v) {
            v *= -learning_rate;
            v /= arma::sqrt(layer_average_of_squares_of_gradients + epsilon);
        });

        return std::move(gradients);
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(learning_rate, beta_1, beta_2, epsilon);
    }

    [[nodiscard]] double getLearningRate() const {
        return learning_rate;
    }

    [[nodiscard]] double getBeta1() const {
        return beta_1;
    }

    [[nodiscard]] double getBeta2() const {
        return beta_2;
    }

    [[nodiscard]] double getEpsilon() const {
        return epsilon;
    }


private:
    double learning_rate;
    double beta_1;
    double beta_2;
    double epsilon;
    mapping_type average_of_squares_of_gradients;
};

CEREAL_REGISTER_TYPE(AdamOptimizer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IOptimizer<double>, AdamOptimizer<double>)
