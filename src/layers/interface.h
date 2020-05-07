#pragma once

#include <vector>
#include <glog/logging.h>

#include <src/tensor.hpp>
#include <src/initializers.hpp>

template<typename T>
struct Gradients {
    Tensor<T> input_gradients;
    Tensor<T> layer_gradients;
};

template<typename T>
class ILayer {
public:
    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual std::string GetName() const = 0;

    [[nodiscard]] virtual Tensor<T> Apply(const Tensor<T> &) const = 0;

    [[nodiscard]] virtual Gradients<T> PullGradientsBackward(
            const Tensor<T> &input,
            const Tensor<T> &output_gradients
    ) const = 0;

    virtual void ApplyGradients(const Tensor<T> &gradients) = 0;

    virtual void SetTrain(bool value) = 0;

    virtual void Initialize(const std::unique_ptr<IInitializer>& initializer) = 0;

    virtual ~ILayer() = default;
};