#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>

struct Gradients {
    arma::mat input_gradients;
    arma::mat layer_gradients;
};

std::string FormatDimensions(const arma::mat &mat) {
    return std::to_string(mat.n_rows) + "x" + std::to_string(mat.n_cols);
}

class ILayer {
public:
    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual std::string GetName() const = 0;

    [[nodiscard]] virtual arma::mat Apply(const arma::mat &) const = 0;

    [[nodiscard]] virtual Gradients PullGradientsBackward(
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const = 0;

    virtual void ApplyGradients(const arma::mat &gradients) = 0;

    virtual ~ILayer() = default;
};

class DenseLayer : public ILayer {
public:
    DenseLayer(arma::uword n_rows, arma::uword n_cols) : weights_and_bias(arma::randu(n_rows + 1, n_cols)) {
    }

    arma::mat GetWeightsAndBias() const {
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
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const override {
        DLOG(INFO) << "Pull gradients for dense layer: "
                   << "input=[" + FormatDimensions(input) + "], "
                   << "output_gradients=[" + FormatDimensions(output_gradients) + "], "
                   << "weights_and_bias=[" + FormatDimensions((weights_and_bias)) + "]";
        auto weights = weights_and_bias.head_rows(weights_and_bias.n_rows - 1);
        auto bias = weights_and_bias.tail_rows(1);
        DLOG(INFO) << "Shape1: [" << FormatDimensions(input.t() * output_gradients) << "], Shape2: ["
                   << FormatDimensions(arma::sum(output_gradients, 0)) << "]";
        return Gradients{
                output_gradients * arma::trans(weights),
                arma::join_cols(input.t() * output_gradients, arma::sum(output_gradients, 0))
        };
    }

    void ApplyGradients(const arma::mat &gradients) override {
        DLOG(INFO) << "Apply gradients for dense layer: "
                   << "gradients[0]=[" + FormatDimensions((gradients)) + "], "
                   << "weights=[" + FormatDimensions((weights_and_bias)) + "]";
        weights_and_bias += gradients;
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
        auto expInput = arma::exp(input);
        return expInput / (expInput + 1);
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const override {
        auto expOutput = arma::exp(output_gradients);
        return Gradients{
                expOutput / arma::square(expOutput + 1),
                arma::mat()
        };
    }

    void ApplyGradients(const arma::mat &) override {}
};