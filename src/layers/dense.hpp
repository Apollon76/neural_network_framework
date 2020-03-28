#pragma once

#include "activations.hpp"
#include "../utils.hpp"
#include <layers.pb.h>
#include <cereal/types/polymorphic.hpp>


class DenseLayer : public ILayer {
public:
    DenseLayer() = default;
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

    template<class Archive>
    void save(Archive& ar) const {
        ar(weights_and_bias.n_rows - 1, weights_and_bias.n_cols);
    }

    template<class Archive>
    void load(Archive & archive) {
        arma::uword rows, cols;
        archive(rows, cols);
        weights_and_bias = DenseLayer(rows, cols).weights_and_bias;
    }

private:
    arma::mat weights_and_bias;
};

CEREAL_REGISTER_TYPE(DenseLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer, DenseLayer)