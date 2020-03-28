#pragma once

#include <src/tensor.hpp>

#include "interface.h"

enum ConvolutionPadding {
    Same,
};

template<typename T>
class Convolution2dLayer : public ILayer<T> {
public:
    Convolution2dLayer(int input_channels, int filters, int kernel_height, int kernel_width, ConvolutionPadding _padding)
            : weights(Tensor<T>::filled(
            {
                    filters,
                    input_channels,
                    kernel_height,
                    kernel_width
            }, arma::fill::randu)),
              padding(_padding) {
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream stream;
        stream << std::endl << weights;
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
                    layer += arma::conv2(input.Values(), weights.Values(), "same");
                }
                result.View().View(batch, filter).Matrix() = layer / weights.D[1];
            }
        }
        return result;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &input,
            const Tensor<T> &output_gradients
    ) const override {
        auto weightsGradients = Tensor<T>(weights.D, arma::fill::zeros);
        auto inputGradients = Tensor<T>(input.D, arma::fill::zeros);
        for (int batch = 0; batch < input.D[0]; batch++) {
            for (int filter = 0; filter < weights.D[0]; filter++) {
                for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                    auto inputImage = input.View().View(batch, input_channel).Matrix();
                    auto outputGradients = output_gradients.View().View(batch, filter).Matrix();
                    for (int w = 0; w < weights.D[2]; w++) {
                        for (int h = 0; h < weights.D[3]; h++) {
                            auto grad = arma::sum(
                                    inputImage.submat(weights.D[2], weights.D[3], inputImage.D[2] - 1,
                                                      inputImage.D[3] - 1) %
                                    outputGradients.submat(outputGradients.D[2] - weights.D[2],
                                                           outputGradients.D[3] - weights.D[3]));
                            weightsGradients.View().View(filter, input_channel).At(w, h) += grad / weights.D[1];
                        }
                    }
                }
            }
        }
        return Gradients<T>{
                inputGradients,
                weightsGradients
        };
    }

    void ApplyGradients(const Tensor<T> &gradients) override {
        for (int filter = 0; filter < weights.D[0]; filter++) {
            for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                weights.View().View(filter, input_channel).Matrix() += gradients.View().View(filter,
                                                                                             input_channel).Matrix();
            }
        }
    }

private:
    Tensor<T> weights;
    ConvolutionPadding padding;
};