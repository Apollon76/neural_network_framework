#pragma once

#include <src/tensor.hpp>
#include <src/utils.hpp>
#include <layers.pb.h>
#include <iostream>
#include <cereal/types/polymorphic.hpp>

#include "activations.hpp"

template<typename T>
class DenseLayer : public ILayer<T> {
public:
    DenseLayer() = default;

    DenseLayer(int input_size, int output_size)
            : weights_and_bias(Tensor<double>({input_size + 1, output_size},
                                              arma::randu(input_size + 1, output_size) -
                                              0.5).ConvertTo<T>()) {
    }

    [[nodiscard]] Tensor<T> GetWeightsAndBias() const {
        return weights_and_bias;
    }

    void SetWeightsAndBias(Tensor<T> new_weights_and_bias) {
        ensure(weights_and_bias.D == new_weights_and_bias.D);
        weights_and_bias = std::move(new_weights_and_bias);
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream stream;
        stream << std::endl << weights_and_bias.ToString();
        return GetName() + stream.str();
    }

    [[nodiscard]] std::string GetName() const override {
        return "Dense[" + FormatDimensions(weights_and_bias) + " (including bias)]";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        ensure(input.D[1] == weights_and_bias.D[0] - 1,
               "Unexpected input size for dense " + std::to_string(input.D[1]) + " != " +
               std::to_string(weights_and_bias.D[0] - 1));
        return Tensor<T>(
                {input.D[0], weights_and_bias.D[1]},
                arma::affmul(weights_and_bias.Values().t(), input.Values().t()).t()
        );
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto weights = weights_and_bias.Values().head_rows(weights_and_bias.D[0] - 1);
        auto bias = weights_and_bias.Values().tail_rows(1);
        return Gradients<T>{
                Tensor<T>(inputs.D, output_gradients.Values() * arma::trans(weights)),
                Tensor<T>(weights_and_bias.D, arma::join_cols(
                        inputs.Values().t() * output_gradients.Values(),
                        arma::sum(output_gradients.Values(), 0))
                )
        };
    }

    void ApplyGradients(const Tensor<T> &gradients) override {
        weights_and_bias.Values() += gradients.Values();
    }

    void SaveWeights(std::ostream *out) {
        DenseWeights matrix;
        matrix.set_n_rows(weights_and_bias.D[0]);
        matrix.set_n_cols(weights_and_bias.D[1]);
        for (int i = 0; i < weights_and_bias.D[0]; ++i) {
            auto row = matrix.add_vectors();
            for (int j = 0; j < weights_and_bias.D[1]; ++j) {
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
    void save(Archive &ar) const {
        ar(weights_and_bias.D[0] - 1, weights_and_bias.D[1]);
    }

    template<class Archive>
    void load(Archive &archive) {
        int rows, cols;
        archive(rows, cols);
        weights_and_bias = DenseLayer(rows, cols).weights_and_bias;
    }

private:
    Tensor<T> weights_and_bias;
};

CEREAL_REGISTER_TYPE(DenseLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, DenseLayer<double>)
