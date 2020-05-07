#pragma once

#include <algorithm>
#include "interface.h"
#include "src/tensor.hpp"

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
            for (arma::uword row = 0; row < v.n_rows; row += kernel_height) {
                for (arma::uword col = 0; col < v.n_cols; col += kernel_width) {
                    int last_row = std::min(v.n_rows - 1, row + kernel_height - 1);
                    int last_col = std::min(v.n_cols - 1, col + kernel_width - 1);
                    arma::uword relative_extremum = v.submat(row, col, last_row, last_col).index_max();
                    auto index_pair = arma::ind2sub(
                            arma::size(last_row - row + 1, last_col - col + 1), relative_extremum
                    );
                    arma::uword extremum = arma::sub2ind(arma::size(v), row + index_pair[0], col + index_pair[1]);
                    auto extremum_value = v(extremum);
                    int x = row / kernel_height;
                    int y = col / kernel_width;
                    extremums.Field().at(a, b, c)(x, y) = extremum;
                    pooled.Field().at(a, b, c)(x, y) = extremum_value;
                }
            }
        });
        return pooled;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &inputs,
            const Tensor<T> &output_gradients
    ) const override {
        auto gradients = Tensor<T>::filled(inputs.D, arma::fill::zeros);
        output_gradients.ForEach([this, &gradients](int a, int b, int c, const arma::Mat<T> &v) {
            auto &result = gradients.Field()(a, b, c);
            for (arma::uword row = 0; row < v.n_rows; row++) {
                for (arma::uword col = 0; col < v.n_cols; col++) {
                    result(extremums.Field()(a, b, c)(row, col)) = v(row, col);
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

private:
    void UpdateDimensions(int &height, int &width) const {
        height = (height + kernel_height - 1) / kernel_height;
        width = (width + kernel_width - 1) / kernel_width;
    }

    int kernel_height, kernel_width;
    mutable Tensor<arma::uword> extremums;
};