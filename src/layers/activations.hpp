#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>
#include "interface.h"
#include <layers.pb.h>


class DenseLayer : public ILayer {
public:
    DenseLayer(arma::uword n_rows, arma::uword n_cols) : weights_and_bias(arma::randu(n_rows + 1, n_cols) - 0.5) {
    }

    [[nodiscard]] arma::mat GetWeightsAndBias() const {
        return weights_and_bias;
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream stream;
        stream << std::endl;
        arma::arma_ostream::print(stream, weights_and_bias, true);
        return GetName() + stream.str();
    }

    [[nodiscard]] std::string GetName() const override {
        return "Dense[" + FormatDimensions(weights_and_bias) + " (including bias)]";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        return arma::affmul(weights_and_bias.t(), input.t()).t();
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &inputs,
            const arma::mat &output_gradients
    ) const override {
        DLOG(INFO) << "Pull gradients for dense layer: "
                   << "inputs=[" + FormatDimensions(inputs) + "], "
                   << "output_gradients=[" + FormatDimensions(output_gradients) + "], "
                   << "weights_and_bias=[" + FormatDimensions((weights_and_bias)) + "]";
        auto weights = weights_and_bias.head_rows(weights_and_bias.n_rows - 1);
        auto bias = weights_and_bias.tail_rows(1);
        return Gradients{
                output_gradients * arma::trans(weights),
                arma::join_cols(inputs.t() * output_gradients, arma::sum(output_gradients, 0))
        };
    }

    void ApplyGradients(const arma::mat &gradients) override {
        DLOG(INFO) << "Apply gradients for dense layer: "
                   << "gradients[0]=[" + FormatDimensions((gradients)) + "], "
                   << "weights=[" + FormatDimensions((weights_and_bias)) + "]";
        weights_and_bias += gradients;
    }

    void SaveWeights(std::ostream *out) {
        DenseWeights matrix;
        matrix.set_n_rows(weights_and_bias.n_rows);
        matrix.set_n_cols(weights_and_bias.n_cols);
        for (arma::uword i = 0; i < weights_and_bias.n_rows; ++i) {
            auto row = matrix.add_vectors();
            for (arma::uword j = 0; j < weights_and_bias.n_cols; ++j) {
                row->add_scalars(weights_and_bias.at(i, j));
            }
        }
        matrix.SerializeToOstream(out);
    }

    void LoadWeights(std::istream *in) {
        DenseWeights matrix;
        matrix.ParseFromIstream(in);
        for (::google::protobuf::uint32 i = 0; i < matrix.n_rows(); ++i) {
            for (::google::protobuf::uint32 j = 0; j < matrix.n_cols(); ++j) {
                weights_and_bias.at(i, j) = matrix.vectors(i).scalars(j);
            }
        }
    }

private:
    arma::mat weights_and_bias;
};

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

    json Serialize() const override {
        return json{
                {"layer_type", "sigmoid_activation"}
        };
    }
};


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

    json Serialize() const override {
        return json{
                {"layer_type", "softmax_activation"}
        };
    }
};


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

    json Serialize() const override {
        return json{
                {"layer_type", "relu_activation"}
        };
    }
};


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
        auto forward_outputs = Apply(inputs);
        auto differentiated = (1 - arma::square(forward_outputs));
        return Gradients{
                output_gradients % (differentiated),
                arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}

    json Serialize() const override {
        return json{
                {"layer_type", "tanh_activation"}
        };
    }
};