#pragma once

#include <src/tensor.hpp>
#include <src/arma_math.hpp>

#include "interface.h"

template<typename T>
class Convolution2dLayer : public ILayer<T> {
public:
    Convolution2dLayer(int input_channels, int filters, int kernel_height, int kernel_width,
                       ConvolutionPadding _padding)
            : weights(Tensor<T>::filled(
            {
                    filters,
                    input_channels,
                    kernel_height,
                    kernel_width
            }, arma::fill::randu)),
              padding(_padding) {
    }

    const Tensor<T> &GetWeights() const {
        return weights;
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream stream;
        stream << std::endl << weights.ToString();
        return GetName() + stream.str();
    }

    [[nodiscard]] std::string GetName() const override {
        return "Conv2d[" + FormatDimensions(weights) + " (including bias)]";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        auto result = Tensor<T>::filled({input.D[0], weights.D[0], input.D[2], input.D[3]}, arma::fill::zeros);
        for (int batch = 0; batch < input.D[0]; batch++) {
            for (int filter = 0; filter < weights.D[0]; filter++) {
                auto layer = arma::Mat<T>(input.D[2], input.D[3], arma::fill::zeros);
                for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                    // todo (sivukhin): use ConvolutionPadding here
                    layer += Conv2d(
                            input.Field()(batch, input_channel),
                            weights.Field()(filter, input_channel),
                            padding
                    );
                }
                result.Field()(batch, filter) = layer / weights.D[1];
            }
        }
        return result;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &input,
            const Tensor<T> &output_gradients
    ) const override {
        auto weightsGradients = Tensor<T>::filled(weights.D, arma::fill::zeros);
        auto inputGradients = Tensor<T>::filled(input.D, arma::fill::zeros);
        for (int batch = 0; batch < input.D[0]; batch++) {
            for (int filter = 0; filter < weights.D[0]; filter++) {
                for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                    auto inputImage = input.Field()(batch, input_channel);
                    auto outputGradients = output_gradients.Field()(batch, filter);
                    auto &currentWeights = weightsGradients.Field()(filter, input_channel);
                    // todo (sivukhin): looks like some kind of convolution...
                    for (int w = 0; w < weights.D[2]; w++) {
                        for (int h = 0; h < weights.D[3]; h++) {
                            currentWeights(w, h) += arma::accu(
                                    outputGradients.submat(0, 0, input.D[2] - w - 1, input.D[3] - h - 1) %
                                    inputImage.submat(w, h, input.D[2] - 1, input.D[3] - 1)
                            ) / weights.D[1];
                        }
                    }
                    inputGradients.Field()(batch, input_channel) += Mirror(Conv2d(
                            Mirror(outputGradients), weights.Field()(filter, input_channel), ConvolutionPadding::Same
                    )) / weights.D[1];
                }
            }
        }
        return Gradients<T>{
                inputGradients,
                weightsGradients
        };
    }

    void ApplyGradients(const Tensor<T> &gradients) override {
        ensure(gradients.D == weights.D);
        for (int filter = 0; filter < weights.D[0]; filter++) {
            for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                weights.Field()(filter, input_channel) += gradients.Field()(filter, input_channel);
            }
        }
    }

private:
    Tensor<T> weights;
    ConvolutionPadding padding;
};