#pragma once

#include <vector>
#include <glog/logging.h>

#include <src/tensor.hpp>
#include "layers_enum.hpp"

template<typename T>
struct Gradients {
    Tensor<T> input_gradients;
    Tensor<T> layer_gradients;
};

template<typename T>
class ILayer {
protected:
    std::string LayerID;
public:

    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual std::string GetName() const = 0;

    [[nodiscard]] virtual Tensor<T> Apply(const Tensor<T> &) const = 0;

    [[nodiscard]] virtual Gradients<T> PullGradientsBackward(
            const Tensor<T> &input,
            const Tensor<T> &output_gradients
    ) const = 0;

    virtual void ApplyGradients(const Tensor<T> &gradients) = 0;

    virtual ~ILayer() = default;

    virtual LayersEnum GetLayerType() const = 0;

    void SetLayerID(std::string name) {
        LayerID = name;
    }

    std::string GetLayerID() {
        return LayerID;
    }

    virtual size_t SaveWeights(std::ostream*) {
        return 0;
    };
};