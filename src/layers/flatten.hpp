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
    FlattenLayer() = default;

    explicit FlattenLayer(TensorDimensions _input_dim) : input_dim(std::move(_input_dim)) {
        ensure(input_dim.size() >= 2, "Can't apply flatten to vector Tensor");
    }

    [[nodiscard]] std::string ToString() const override {
        std::string dims;
        for (auto d : input_dim) {
            if (!dims.empty()) {
                dims += ", ";
            }
            dims += std::to_string(d);
        }
        return GetName() + ", input dimensions: " + FormatDimensions(input_dim);
    }

    [[nodiscard]] std::string GetName() const override {
        return "Flatten";
    }

    [[nodiscard]] Tensor<T> Apply(const Tensor<T> &input) const override {
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
        auto rows = output_gradients.D[0];
        std::vector<int> current_dim = input_dim;
        current_dim[0] = rows;
        int h = input_dim[input_dim.size() - 2];
        int w = input_dim[input_dim.size() - 1];
        auto tensor = Tensor<T>(
                current_dim,
                createValuesContainer<T>(current_dim)
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

    template<class Archive>
    void serialize(Archive &ar) {
        ar(input_dim);
    }

private:
    TensorDimensions input_dim;
};

CEREAL_REGISTER_TYPE(FlattenLayer<double>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILayer<double>, FlattenLayer<double>)
