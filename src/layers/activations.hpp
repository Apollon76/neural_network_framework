#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>
#include <cereal/types/polymorphic.hpp>

#include "interface.h"

class SigmoidActivationLayer : public ILayer {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "SigmoidActivation";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        return 1 / (1 + arma::exp(-input));
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &inputs,
            const arma::mat &output_gradients
    ) const override {
        auto activation_result = Apply(inputs);
        return Gradients{
                output_gradients % ((activation_result) % (1 - activation_result)),
                arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}

    template<class Archive>
    void serialize(Archive&) {}
};
CEREAL_REGISTER_TYPE(SigmoidActivationLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer, SigmoidActivationLayer)


class SoftmaxActivationLayer : public ILayer {
 public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "SoftmaxActivation";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        arma::mat shifted_input = input - arma::max(input, 1) * arma::ones(1, input.n_cols); // todo (mpivko): do we need shift?
        arma::mat repeated_sum = arma::sum(arma::exp(shifted_input), 1) * arma::ones(1, input.n_cols);
        return arma::exp(shifted_input) / repeated_sum;
    }

    [[nodiscard]] Gradients PullGradientsBackward(
        const arma::mat &inputs,
        const arma::mat &output_gradients
    ) const override {
        arma::mat forward_outputs = Apply(inputs); // todo (mpivko): maybe cache this in field?

        arma::mat sum = arma::sum(output_gradients % forward_outputs, 1);
        arma::mat repeated_sum = sum * arma::ones(1, output_gradients.n_cols);

        return Gradients{
            (output_gradients % forward_outputs) - (repeated_sum % forward_outputs),
            arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}

    template<class Archive>
    void serialize(Archive&) {}
};
CEREAL_REGISTER_TYPE(SoftmaxActivationLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer, SoftmaxActivationLayer)


class ReLUActivationLayer : public ILayer {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "ReLU Activation";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        auto result = input;
        result.for_each([](arma::mat::elem_type& value) {
            if (value < 0)
                value = 0;
        });
        return result;
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &inputs,
            const arma::mat &output_gradients
    ) const override {
        auto differentiated = inputs;
        differentiated.for_each([](arma::mat::elem_type& value) {
            if (value < 0)
                value = 0;
            else
                value = 1;
        });
        return Gradients{
                output_gradients % differentiated,
                arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}

    template<class Archive>
    void serialize(Archive&) {}
};
CEREAL_REGISTER_TYPE(ReLUActivationLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer, ReLUActivationLayer)


class TanhActivationLayer : public ILayer {
public:
    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }
    [[nodiscard]] std::string GetName() const override {
        return "Tanh Activation";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        return (arma::exp(input) - arma::exp(-input)) / (arma::exp(input) + arma::exp(-input));
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &inputs,
            const arma::mat &output_gradients
    ) const override {
        arma::mat forward_outputs = Apply(inputs);
        arma::mat differentiated = (1 - arma::square(forward_outputs));
        return Gradients{
                output_gradients % (differentiated),
                arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}

    template<class Archive>
    void serialize(Archive&) {}
};
CEREAL_REGISTER_TYPE(TanhActivationLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer, TanhActivationLayer)