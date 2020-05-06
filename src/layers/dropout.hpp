#pragma once

#include "interface.h"

#include <armadillo>
#include <cereal/types/polymorphic.hpp>
#include <src/tensor.hpp>


template<typename T>
class DropoutLayer : public ILayer<T> {
public:
    DropoutLayer() = default;

    DropoutLayer(double _p): p(_p), distr(0.0, 1.0), engine(0) {}

    [[nodiscard]] std::string ToString() const override {
        return GetName() + ": p=" + std::to_string(p);
    }

    [[nodiscard]] std::string GetName() const override {
        return "Dropout";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) override {
        mask = Tensor<T>::filled(input.D, arma::fill::zeros);
        mask.ForEach([&](int, int, int, arma::Mat<T> &data){
            data.imbue([&]() {
                if (distr(engine) > p) {
                    return 1;
                } else {
                    return 0;
                }
            });
        });
        auto result = Tensor<T>(input.D, input.Field());
        result.ForEach([&](int a, int b, int c, arma::Mat<T> &data){
            data %= mask.Field().at(a, b, c) / (1 - p);
        });
        return result;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) override {
        auto result = Tensor<T>(output_gradients.D, output_gradients.Field());
        result.ForEach([&](int a, int b, int c, arma::Mat<T> &data){
            data %= mask.Field().at(a, b, c);
        });
        return Gradients<T>{
                result,
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &gradients) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(p);
    }

private:
    double p;
    Tensor<T> mask;
    std::mt19937 engine;
    std::uniform_real_distribution<double> distr;
};

CEREAL_REGISTER_TYPE(DropoutLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, DropoutLayer<double>)
