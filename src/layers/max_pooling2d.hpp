#pragma once

#include <algorithm>
#include "interface.h"
#include "src/tensor.hpp"
#include "src/initializers.hpp"

template<typename T>
class MaxPooling2dLayer : public ILayer<T> {
public:
    MaxPooling2dLayer() = default;

    MaxPooling2dLayer(int _kernel_height, int _kernel_width)
            : kernel_height(_kernel_height), kernel_width(_kernel_width), extremums() {}

    [[nodiscard]] std::string ToString() const override {
        return GetName() + ": " + std::to_string(kernel_height) + "x" + std::to_string(kernel_width);
    }

    [[nodiscard]] std::string GetName() const override {
        return "MaxPooling";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        auto new_dimensions = input.D;
        ensure(new_dimensions.size() >= 3, "input must have at least three dimensions");
        UpdateDimensions(new_dimensions[new_dimensions.size() - 2], new_dimensions[new_dimensions.size() - 1]);

        auto pooled = Tensor<T>::filled(new_dimensions, arma::fill::zeros);
        extremums = Tensor<arma::uword>::filled(new_dimensions, arma::fill::zeros);
        input.ForEach([this, &pooled](int a, int b, int c, const arma::Mat<T> &v) {
            auto& currPooled = pooled.Field().at(a, b, c);
            auto& currExtremum = extremums.Field().at(a, b, c);
            for (arma::uword row = 0, row_idx = 0; row < currPooled.n_rows; row++, row_idx += kernel_height) {
                for (arma::uword col = 0, col_idx = 0; col < currPooled.n_cols; col++, col_idx += kernel_width) {
                    currExtremum(row, col) = v(
                            arma::span(row_idx, row_idx + kernel_height - 1),
                            arma::span(col_idx, col_idx + kernel_height - 1)).index_max();
                    currPooled(row, col) = v(
                            arma::span(row_idx, row_idx + kernel_height - 1),
                            arma::span(col_idx, col_idx + kernel_height - 1)).max();
                }
            }
        });
        return pooled;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        ensure(output_gradients.D == extremums.D);
        auto gradients = Tensor<T>::filled(inputs.D, arma::fill::zeros);
        output_gradients.ForEach([this, &gradients](int a, int b, int c, const arma::Mat<T> &v) {
            auto &result = gradients.Field()(a, b, c);
            for (arma::uword row = 0; row < v.n_rows; row++) {
                for (arma::uword col = 0; col < v.n_cols; col++) {
                    result.submat(row * kernel_height, col * kernel_height, (row + 1) * kernel_height - 1,
                                  (col + 1) * kernel_height - 1)(extremums.Field()(a, b, c)(row, col)) = v(row, col);
                }
            }
        });
        return Gradients<T>{
                gradients,
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

    void SetTrain(bool) override {}

    template<class Archive>
    void serialize(Archive &ar) {
        ar(kernel_height);
        ar(kernel_width);
    }

    void Initialize(const std::unique_ptr<IInitializer<T>>& initializer) override {
        std::ignore = initializer;
    };

private:
    void UpdateDimensions(int &height, int &width) const {
        height = height / kernel_height;
        width = width / kernel_width;
    }

    int kernel_height, kernel_width;
    mutable Tensor<arma::uword> extremums;
};