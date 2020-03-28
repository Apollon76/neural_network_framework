#pragma once

#include <armadillo>
#include <src/utils.hpp>
#include <src/layers/interface.h>
#include <src/tensor.hpp>

template<typename T>
class IOptimizer : public ISerializable {
public:
    [[nodiscard]] virtual Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) = 0;

    virtual ~IOptimizer() = default;
};

template<typename T>
class Optimizer : public IOptimizer<T> {
public:
    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        UNUSED(layer)
        return Tensor<T>(gradients.D, -gradients.Values() * learning_rate);
    }

    [[nodiscard]] json Serialize() const override {
        return {
                {"optimizer_type", "optimizer"},
                {"params",         {{"learning_rate", learning_rate}}}
        };
    }

private:
    double learning_rate;
};

template<typename T>
class MomentumOptimizer : public IOptimizer<T> {
public:
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

    [[nodiscard]] json Serialize() const override {
        return {
                {"optimizer_type", "momentum"},
                {"params",         {{"learning_rate", learning_rate}, {"momentum", momentum}}}
        };
    }

private:
    double learning_rate;
    double momentum;

    std::unordered_map<const ILayer<T> *, Tensor<T>> previous_values;
};

template<typename T>
class RMSPropOptimizer : public IOptimizer<T> {
public:
    explicit RMSPropOptimizer(double _learning_rate = 0.001, double _rho = 0.9, double _eps = 1e-7)
            : learning_rate(_learning_rate), rho(_rho), epsilon(_eps) {
    }

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        auto it = previous_mean.find(layer);
        arma::Mat<T> previous_gradient;
        if (it == previous_mean.end()) {
            previous_gradient.zeros(arma::size(gradients.Values()));
        } else {
            previous_gradient = it->second;
        }

        auto currentMean = previous_mean[layer]
                                   = rho * previous_gradient + (1 - rho) * arma::square(gradients.Values());
        return Tensor<T>(gradients.D, -learning_rate * (gradients.Values() / arma::sqrt(currentMean + epsilon)));
    }

    [[nodiscard]] json Serialize() const override {
        return {
                {"optimizer_type", "rmsprop"},
                {"params",         {{"learning_rate", learning_rate}, {"rho", rho}, {"eps", epsilon}}}
        };
    }

private:
    double learning_rate;
    double rho;
    double epsilon;

    std::unordered_map<const ILayer<T> *, arma::Mat<T>> previous_mean;
};

template<typename T>
class AdamOptimizer : public IOptimizer<T> {
    using mapping_type = std::unordered_map<const ILayer<T> *, arma::Mat<T>>;
public:
    explicit AdamOptimizer(double _learning_rate = 0.001, double _beta_1 = 0.9, double _beta_2 = 0.999,
                           double _eps = 1e-7)
            : learning_rate(_learning_rate), beta_1(_beta_1), beta_2(_beta_2), epsilon(_eps) {
    }

    [[nodiscard]] Tensor<T> GetGradientStep(const Tensor<T> &gradients, const ILayer<T> *layer) override {
        auto previous_gradient_avg = GetOrCreatePrevious(layer, average_of_gradients, arma::size(gradients.Values()));
        auto previous_square_gradient_avg = GetOrCreatePrevious(layer, average_of_squares_of_gradients,
                                                                arma::size(gradients.Values()));

        auto cur_avg = average_of_gradients[layer] = beta_1 * previous_gradient_avg + (1 - beta_1) * gradients.Values();
        auto cur_square_avg = average_of_squares_of_gradients[layer] =
                                      beta_2 * previous_square_gradient_avg +
                                      (1 - beta_2) * arma::square(gradients.Values());

        return Tensor<T>(gradients.D, -learning_rate * gradients.Values() / arma::sqrt(cur_square_avg + epsilon));
    }

    [[nodiscard]] json Serialize() const override {
        return {
                {"optimizer_type", "adam"},
                {"params",         {{"learning_rate", learning_rate}, {"beta_1", beta_1}, {"beta_2", beta_2}, {"eps", epsilon}}}
        };
    }

private:
    arma::Mat<T> GetOrCreatePrevious(const ILayer<T> *layer, mapping_type &mapping, arma::SizeMat size) {
        auto it = mapping.find(layer);
        arma::mat previous;
        if (it != mapping.end()) {
            return it->second;
        }
        previous.zeros(size);
        return previous;
    }

private:
    double learning_rate;
    double beta_1;
    double beta_2;
    double epsilon;
    mapping_type average_of_gradients;
    mapping_type average_of_squares_of_gradients;
};
