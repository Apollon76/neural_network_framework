#pragma once

#include "activations.hpp"
#include "../utils.hpp"


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

    [[nodiscard]] arma::mat Apply(const arma::mat& input) const override {
        return arma::affmul(weights_and_bias.t(), input.t()).t();
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat& inputs,
            const arma::mat& output_gradients
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

    void ApplyGradients(const arma::mat& gradients) override {
        DLOG(INFO) << "Apply gradients for dense layer: "
                   << "gradients[0]=[" + FormatDimensions((gradients)) + "], "
                   << "weights=[" + FormatDimensions((weights_and_bias)) + "]";
        weights_and_bias += gradients;
    }

    json Serialize() const override {
        return json{
                {"layer_type", "dense"},
                {"params",     {
                                       {"n_rows", weights_and_bias.n_rows - 1},
                                       {"n_cols", weights_and_bias.n_cols}
                               }
                }
        };
    }

private:
    arma::mat weights_and_bias;
};