#pragma once

#include <armadillo>
#include <src/tensor.hpp>

template<typename T>
class ILoss : public ISerializable {
public:
    [[nodiscard]] virtual double GetLoss(const Tensor<T> &inputs, const Tensor<T> &outputs) const = 0;

    [[nodiscard]] virtual Tensor<T> GetGradients(const Tensor<T> &input, const Tensor<T> &outputs) const = 0;

    virtual ~ILoss() = default;
};

template<typename T>
class MSELoss : public ILoss<T> {
public:
    [[nodiscard]] double GetLoss(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return arma::accu(arma::pow(inputs.Values() - outputs.Values(), 2)) / inputs.BatchCount();
    }

    [[nodiscard]] Tensor<T> GetGradients(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return Tensor<T>(inputs.D, 2 * (inputs.Values() - outputs.Values()) / inputs.BatchCount());
    }

    [[nodiscard]] json Serialize() const override {
        return {"loss_type", "mse"};
    }
};

template<typename T>
class CategoricalCrossEntropyLoss : public ILoss<T> {
public:
    [[nodiscard]] double GetLoss(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return arma::mean(arma::sum(outputs.Values() % arma::log(inputs.Values()), 1) * -1);
    }

    [[nodiscard]] Tensor<T> GetGradients(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return Tensor<T>(inputs.D, -outputs.Values() / inputs.Values());
    }

    [[nodiscard]] json Serialize() const override {
        return {"loss_type", "categorical_cross_entropy"};
    }
};