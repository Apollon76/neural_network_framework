#pragma once

#include <armadillo>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include <cereal/types/polymorphic.hpp>
#include <src/tensor.hpp>

#include "interface.h"

template<typename T>
class FlattenLayer : public ILayer<T> {
public:
    explicit FlattenLayer(TensorDimensions _input_dim) : input_dim(std::move(_input_dim)) {
        ensure(input_dim.size() >= 2, "Can't apply flatten to vector Tensor");
    }

    [[nodiscard]] std::string ToString() const override {
        return GetName();
    }

    [[nodiscard]] std::string GetName() const override {
        return "Flatten";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
        ensure(input.D == input_dim, "Unexpected input dimension in flatten layer");
        auto rows = input.D[0];
        auto cols = std::accumulate(input.D.begin() + 1, input.D.end(), 1, std::multiplies<T>());
        auto tensor = Tensor<T>(
                {rows, cols},
                arma::field<arma::Mat<T>>({arma::Mat<T>(rows, cols, arma::fill::zeros)})
        );
        auto &matrix = tensor.Field()(0);
        auto offsets = std::vector<int>(rows);
        input.ForEach([&matrix, &offsets](int a, int, int, const arma::Mat<T> &v) {
            matrix.submat(a, offsets[a], a, offsets[a] + v.n_elem - 1) = arma::vectorise(v, 1);
            offsets[a] += v.n_elem;
        });
        return tensor;
    }

    [[nodiscard]] Gradients<T> PullGradientsBackward(
            const Tensor<T> &,
            const Tensor<T> &output_gradients
    ) const override {
        ensure(input_dim[0] == output_gradients.D[0], "Batch dimensions doesn't match for flatten layer");
        auto rows = input_dim[0];
        int h = input_dim[input_dim.size() - 2];
        int w = input_dim[input_dim.size() - 1];
        auto tensor = Tensor<T>(
                input_dim,
                createValuesContainer<T>(input_dim)
        );
        auto offsets = std::vector<int>(rows);
        tensor.ForEach([h, w, &offsets, &output_gradients](int a, int, int, arma::Mat<T> &v) {
            v = arma::Mat<T>(h, w);
            for (int i = 0; i < h; i++) {
                v.submat(i, 0, i, w - 1) = output_gradients.Values().submat(a, offsets[a], a, offsets[a] + w - 1);
                offsets[a] += w;
            }
        });
        return Gradients<T>{
                tensor,
                Tensor<T>()
        };
    }

    void ApplyGradients(const Tensor<T> &) override {}

private:
    TensorDimensions input_dim;
};