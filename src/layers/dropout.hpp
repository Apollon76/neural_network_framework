#pragma once

#include "interface.h"

#include <armadillo>
#include <cereal/types/polymorphic.hpp>
#include <src/tensor.hpp>


template<typename T>
class DropoutLayer : public ILayer<T> {
public:
    DropoutLayer() = default;

    DropoutLayer(double _p): p(_p) {}

    [[nodiscard]] std::string ToString() const override {
        return GetName() + ": p=" + std::to_string(p);
    }

    [[nodiscard]] std::string GetName() const override {
        return "Dropout";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        mask = Tensor<T>::filled(input.D, arma::fill::randu);
        mask.ForEach([&](int, int, int, arma::Mat<T> &data){
            data.transform([&](T value) { return value > p ? 1 : 0; });
        });
        auto result = Tensor<T>(input.D, input.Field());
        result.ForEach([&](int a, int b, int c, arma::Mat<T> &data){
            data %= mask.Field().at(a, b, c) / (1 - p);
        });
        return result;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T>&,
            const Tensor<T> &output_gradients
    ) const override {
        auto result = Tensor<T>(output_gradients.D, output_gradients.Field());
        result.ForEach([&](int a, int b, int c, arma::Mat<T> &data){
            data %= mask.Field().at(a, b, c);
        });
        return Gradients<T>{
                result,
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T>&) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(p);
    }

private:
    double p;
    mutable Tensor<T> mask;
};

CEREAL_REGISTER_TYPE(DropoutLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, DropoutLayer<double>)
