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
            : weights_and_bias({input_size + 1, output_size}, arma::randu(input_size + 1, output_size) - 0.5) {
    }

    [[nodiscard]] Tensor<T> GetWeightsAndBias() const {
        return weights_and_bias;
    }

    inline LayersEnum GetLayerType() const override {
        return LayersEnum::DENSE;
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
        return Tensor<T>(
                {input.D[0], weights_and_bias.D[1]},
                arma::affmul(weights_and_bias.Values().t(), input.Values().t()).t()
        );
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        DLOG(INFO) << "Pull gradients for dense layer: "
                   << "inputs=[" + FormatDimensions(inputs) + "], "
                   << "output_gradients=[" + FormatDimensions(output_gradients) + "], "
                   << "weights_and_bias=[" + FormatDimensions((weights_and_bias)) + "]";
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
        DLOG(INFO) << "Apply gradients for dense layer: "
                   << "gradients[0]=[" + FormatDimensions(gradients) + "], "
                   << "weights=[" + FormatDimensions(weights_and_bias) + "]";
        weights_and_bias.Values() += gradients.Values();
    }

    size_t SaveWeights(std::ostream *out) override {
        DenseWeights matrix;
        matrix.set_name(this->LayerID);
        matrix.set_n_rows(weights_and_bias.D[0]);
        matrix.set_n_cols(weights_and_bias.D[1]);
        for (int i = 0; i < weights_and_bias.D[0]; ++i) {
            auto row = matrix.add_vectors();
            for (int j = 0; j < weights_and_bias.D[1]; ++j) {
                row->add_scalars(weights_and_bias.at(i, j));
            }
        }
        matrix.SerializeToOstream(out);
        return matrix.ByteSizeLong();
    }

    void LoadWeights(std::istream *in) {
        DenseWeights matrix;
        matrix.ParseFromIstream(in);
        for (::google::protobuf::uint32 i = 0; i < matrix.n_rows(); ++i) {
            for (::google::protobuf::uint32 j = 0; j < matrix.n_cols(); ++j) {
                weights_and_bias.at(i, j) = matrix.vectors(i).scalars(j);
            }
        }
        this->LayerID = matrix.name();
    }

    void LoadWeights(const DenseWeights &matrix) {
        for (::google::protobuf::uint32 i = 0; i < matrix.n_rows(); ++i) {
            for (::google::protobuf::uint32 j = 0; j < matrix.n_cols(); ++j) {
                weights_and_bias.at(i, j) = matrix.vectors(i).scalars(j);
            }
        }
        this->LayerID = matrix.name();
    }

    template<class Archive>
    void save(Archive &ar) const {
        ar(weights_and_bias.D[0] - 1, weights_and_bias.D[1], this->LayerID);
    }

    template<class Archive>
    void load(Archive &archive) {
        int rows, cols;
        std::string LayerID;
        archive(rows, cols, LayerID);
        weights_and_bias = DenseLayer(rows, cols).weights_and_bias;
        this->SetLayerID(LayerID);
    }



private:
    Tensor<T> weights_and_bias;
};

CEREAL_REGISTER_TYPE(DenseLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, DenseLayer<double>)
