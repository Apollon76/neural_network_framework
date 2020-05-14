#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>
#include <cereal/types/polymorphic.hpp>
#include <src/tensor.hpp>
#include <src/initializers.hpp>

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
        return input.template Transform<T>([](const arma::Mat<T> &v) {
            arma::Mat<T> value = 1 / (1 + arma::exp(-v));
            return value;
        });
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto activation = Apply(inputs);
        return Gradients<T>{
                output_gradients.template DiffWith<T>(activation, [](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                    arma::Mat<T> value = a % (b % (1 - b));
                    return value;
                }),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    void SetTrain(bool) override {}

    void Initialize(const std::unique_ptr<IInitializer<T>>& initializer) override {
        std::ignore = initializer;
    };

    template<class Archive>
    void serialize(Archive &) {}
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
                arma::max(input.Values(), 1) * arma::ones<arma::Mat<T>>(1, input.D[1]);
        arma::Mat<T> repeated_sum = arma::sum(arma::exp(shifted_input), 1) * arma::ones<arma::Mat<T>>(1, input.D[1]);
        return Tensor<T>(input.D, arma::exp(shifted_input) / repeated_sum);
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto forward_outputs = Apply(inputs); // todo (mpivko): maybe cache this in field?

        arma::Mat<T> sum = arma::sum(output_gradients.Values() % forward_outputs.Values(), 1);
        arma::Mat<T> repeated_sum = sum * arma::ones<arma::Mat<T>>(1, output_gradients.D[1]);

        return Gradients<T>{
                Tensor<T>(inputs.D,
                          (output_gradients.Values() % forward_outputs.Values()) -
                          (repeated_sum % forward_outputs.Values())),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    void SetTrain(bool) override {}

    template<class Archive>
    void serialize(Archive &) {}

    void Initialize(const std::unique_ptr<IInitializer<T>>& initializer) override {
        std::ignore = initializer;
    };
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
        return input.template Transform<T>([](const arma::Mat<T> &v) -> arma::Mat<T> {
            arma::Mat<T> result = v;
            result.for_each([](T &value) {
                if (value < 0) {
                    value = 0;
                }
            });
            return result;
        });
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        return Gradients<T>{
                output_gradients.template DiffWith<T>(inputs, [](const arma::Mat<T> &a, const arma::Mat<T> &b) -> arma::Mat<T> {
                    arma::Mat<T> diff = b;
                    diff.for_each([](T &value) {
                        if (value < 0) {
                            value = 0;
                        } else {
                            value = 1;
                        }
                    });
                    return a % diff;
                }),
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    void SetTrain(bool) override {}

    template<class Archive>
    void serialize(Archive &) {}

    void Initialize(const std::unique_ptr<IInitializer<T>>& initializer) override {
        std::ignore = initializer;
    };
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

    void SetTrain(bool) override {}

    template<class Archive>
    void serialize(Archive &) {}

    void Initialize(const std::unique_ptr<IInitializer<T>>& initializer) override {
        std::ignore = initializer;
    };
};
CEREAL_REGISTER_TYPE(TanhActivationLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, TanhActivationLayer<double>)