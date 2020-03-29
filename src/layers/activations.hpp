#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>
#include <cereal/types/polymorphic.hpp>
#include <src/tensor.hpp>

#include "interface.h"

template<typename T>
class SigmoidActivationLayer : public ILayer<T> {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "SigmoidActivation";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        return Tensor<T>(input.D, 1 / (1 + arma::exp(-input.Values())));
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto activation_result = Apply(inputs);
        return Gradients<T>{
                Tensor<T>(
                        inputs.D,
                        output_gradients.Values() % ((activation_result.Values()) % (1 - activation_result.Values()))
                ),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(this->LayerID);
    }

    inline LayersEnum GetLayerType() const override {
        return LayersEnum::SIGMOID_ACTIVATION;
    }
};
CEREAL_REGISTER_TYPE(SigmoidActivationLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, SigmoidActivationLayer<double>)


template<typename T>
class SoftmaxActivationLayer : public ILayer<T> {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "SoftmaxActivation";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        // todo (sivukhin): Generalize sotfmax for tensor of arbitary dimension
        ensure(input.Rank() == 2, "SoftMax activation supported only for tensors of rank = 2");
        arma::Mat<T> shifted_input =
                input.Values() -
                arma::max(input.Values(), 1) * arma::ones(1, input.D[1]); // todo (mpivko): do we need shift?
        arma::Mat<T> repeated_sum = arma::sum(arma::exp(shifted_input), 1) * arma::ones(1, input.D[1]);
        return Tensor<T>(input.D, arma::exp(shifted_input) / repeated_sum);
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto forward_outputs = Apply(inputs); // todo (mpivko): maybe cache this in field?

        arma::mat sum = arma::sum(output_gradients.Values() % forward_outputs.Values(), 1);
        arma::mat repeated_sum = sum * arma::ones(1, output_gradients.D[1]);

        return Gradients<T>{
                Tensor<T>(inputs.D,
                          (output_gradients.Values() % forward_outputs.Values()) -
                          (repeated_sum % forward_outputs.Values())),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    inline LayersEnum GetLayerType() const override {
        return LayersEnum::SOFTMAX_ACTIVATION;
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(this->LayerID);
    }
};
CEREAL_REGISTER_TYPE(SoftmaxActivationLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, SoftmaxActivationLayer<double>)


template<typename T>
class ReLUActivationLayer : public ILayer<T> {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "ReLU Activation";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        auto result = input.Values();
        result.for_each([](arma::mat::elem_type &value) {
            if (value < 0)
                value = 0;
        });
        return Tensor<T>(input.D, result);
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto differentiated = inputs.Values();
        differentiated.for_each([](T &value) {
            if (value < 0)
                value = 0;
            else
                value = 1;
        });
        return Gradients<T>{
                Tensor<T>(inputs.D, output_gradients.Values() % differentiated),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(this->LayerID);
    }

    inline LayersEnum GetLayerType() const override {
        return LayersEnum::RELU_ACTIVATION;
    }
};
CEREAL_REGISTER_TYPE(ReLUActivationLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, ReLUActivationLayer<double>)


template<typename T>
class TanhActivationLayer : public ILayer<T> {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "Tanh Activation";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        auto &values = input.Values();
        return Tensor<T>(
                input.D,
                (arma::exp(values) - arma::exp(-values)) / (arma::exp(values) + arma::exp(-values))
        );
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        arma::Mat<T> forward_outputs = Apply(inputs).Values();
        arma::Mat<T> differentiated = (1 - arma::square(forward_outputs));
        return Gradients<T>{
                Tensor<T>(inputs.D, output_gradients.Values() % (differentiated)),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(this->LayerID);
    }

    inline LayersEnum GetLayerType() const override {
        return LayersEnum::TANH_ACTIVATION;
    }
};
CEREAL_REGISTER_TYPE(TanhActivationLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, TanhActivationLayer<double>)