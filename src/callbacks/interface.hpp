#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma once

#include <vector>
#include <optional>
#include <functional>

#include "src/tensor.hpp"
#include "src/neural_network_interface.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"

enum CallbackSignal {
    Continue,
    Stop,
};

template<typename T>
class ANeuralNetworkCallback {
public:
    virtual std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *nn, int epoch) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void()>>
    FitBatch(const nn_framework::data_processing::Data<T> &batch_data, int batch_id, int batches_count) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void(const Tensor<T> &output)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &input) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void(const Gradients<T> &gradients)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &input, const Tensor<T> &output_gradients) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void(const Tensor<T> &output_gradients)>>
    OptimizerGradients(const Tensor<T> &output) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void(const Tensor<T> &gradient_step)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &layer_gradients) {
        return std::nullopt;
    }

    virtual std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &gradients) {
        return std::nullopt;
    }

    virtual ~ANeuralNetworkCallback() = default;
};

#pragma clang diagnostic pop