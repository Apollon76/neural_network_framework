#pragma once

#include <src/tensor.hpp>

#include "interface.h"

enum ConvolutionPadding {
    Valid,
    Same,
};

template<typename T>
class Convolution2d : public ILayer<T> {
public:
    Convolution2d(int input_channels, int filters, int kernel_width, int kernel_height, ConvolutionPadding _padding)
            : weights(Tensor<T>::filled(
            {
                    filters,
                    input_channels,
                    kernel_width,
                    kernel_height
            }, arma::fill::randu)),
              biases(Tensor<T>::filled({input_channels}, arma::fill::randu)),
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
                    layer += arma::conv2(input.Values(), weights.Values(), "same") + biases.at(0, input_channel);
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
//        auto biasGradients = arma::sum(output_gradients.Values(), 0) / weights.D[1];
        auto weightsGradients = Tensor<T>(weights.D, arma::fill::zeros);
        for (int batch = 0; batch < input.D[0]; batch++) {
            for (int filter = 0; filter < weights.D[0]; filter++) {
                for (int input_channel = 0; input_channel < weights.D[1]; input_channel++) {
                    auto inputImage = input.View().View(batch, input_channel).Matrix();
                    auto outputGradients = output_gradients.View().View(batch, filter).Matrix();
                    for (int w = 0; w < weights.D[2]; w++) {
                        for (int h = 0; h < weights.D[3]; h++) {
                            weightsGradients.View()
                        }
                    }
                }
            }
        }
    }

    void ApplyGradients(const Tensor<T> &gradients) override {

    }

private:
    Tensor<T> weights;
    Tensor<T> biases;
    ConvolutionPadding padding;
};