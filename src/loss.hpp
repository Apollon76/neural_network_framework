#pragma once

#include <armadillo>
#include <cereal/types/polymorphic.hpp>

#include <src/tensor.hpp>

template<typename T>
class ILoss {
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

    template<class Archive>
    void serialize(Archive&) {}
};

CEREAL_REGISTER_TYPE(MSELoss<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILoss<double>, MSELoss<double>)

template<typename T>
class CategoricalCrossEntropyLoss : public ILoss<T> {
public:
    [[nodiscard]] double GetLoss(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return arma::mean(arma::sum(outputs.Values() % arma::log(inputs.Values()), 1) * -1);
    }

    [[nodiscard]] Tensor<T> GetGradients(const Tensor<T> &inputs, const Tensor<T> &outputs) const override {
        return Tensor<T>(inputs.D, -outputs.Values() / inputs.Values());
    }

    template<class Archive>
    void serialize(Archive&) {}
};

CEREAL_REGISTER_TYPE(CategoricalCrossEntropyLoss<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILoss<double>, CategoricalCrossEntropyLoss<double>)