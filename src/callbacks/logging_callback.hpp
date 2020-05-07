#pragma once

#include <string>

#include "interface.hpp"
#include "src/neural_network_interface.hpp"

enum LoggingLevelFlags {
    Epoch = 1,
    Batch = 2,
    Internals = 4,
};

inline LoggingLevelFlags operator|(LoggingLevelFlags a, LoggingLevelFlags b) {
    return static_cast<LoggingLevelFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool HasFlag(LoggingLevelFlags collection, LoggingLevelFlags flag) {
    return (static_cast<uint32_t>(collection) & static_cast<uint32_t>(flag)) != 0;
}

template<typename T>
class LoggingCallback : public ANeuralNetworkCallback<T> {
public:
    explicit LoggingCallback(LoggingLevelFlags _flags = LoggingLevelFlags::Epoch | LoggingLevelFlags::Batch)
            : indentation(0), flags(_flags) {}

    std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *, int epoch) override {
        if (!HasFlag(flags, LoggingLevelFlags::Epoch)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start epoch " << epoch;
        indentation++;
        return [epoch, this](const Tensor<T> &, double loss) {
            indentation--;
            LOG(INFO) << Indentation() << "Finish epoch " << epoch << " with loss " << loss;
            return CallbackSignal::Continue;
        };
    }

    std::optional<std::function<void()>>
    FitBatch(const nn_framework::data_processing::Data<T> &batch_data, int batch_id, int batches_count) override {
        if (!HasFlag(flags, LoggingLevelFlags::Batch)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start fitting batch " << batch_id << " / " << batches_count << " "
                  << "with input tensor " << FormatTensor(batch_data.input) << " "
                  << "and output tensor " << FormatTensor(batch_data.output);
        indentation++;
        return [this]() {
            indentation--;
            LOG(INFO) << Indentation() << "Finish fitting batch";
        };
    }

    std::optional<std::function<void(const Tensor<T> &output)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &input) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start forward pass for layer " << layer->GetName() << " "
                  << "with input tensor " << FormatTensor(input);
        return [this](const Tensor<T> &output) {
            LOG(INFO) << Indentation() << "Finish forward pass with output tensor " << FormatTensor(output);
        };
    }

    std::optional<std::function<void(const Gradients<T> &gradients)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &input, const Tensor<T> &output_gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation() << "Start backward pass for layer " << layer->GetName()
                  << " with input tensor " << FormatTensor(input)
                  << " and output gradients tensor " << FormatTensor(output_gradients);
        return [this](const Gradients<T> &gradients) {
            LOG(INFO) << Indentation() << "Finish backward pass"
                      << " with input gradient tensor " << FormatTensor(gradients.input_gradients)
                      << " and layer gradient tensor " << FormatTensor(gradients.layer_gradients);
        };
    }

    std::optional<std::function<void(const Tensor<T> &output_gradients)>>
    OptimizerGradients(const Tensor<T> &output) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start calculating output gradients for output tensor " << FormatTensor(output);
        return [this](const Tensor<T> &gradients) {
            LOG(INFO) << Indentation() << "Finish calculating output gradients tensor " << FormatTensor(gradients);
        };
    }

    std::optional<std::function<void(const Tensor<T> &gradient_step)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &layer_gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start calculating gradient step for layer " << layer->GetName() << " "
                  << "with layer gradients tensor " << FormatTensor(layer_gradients);
        return [this](const Tensor<T> &step) {
            LOG(INFO) << Indentation() << "Finish calculating gradient step tensor " << FormatTensor(step);
        };
    }

    std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        LOG(INFO) << Indentation()
                  << "Start applying gradients for layer " << layer->GetName() << " "
                  << "with layer gradients tensor " << FormatTensor(gradients);
        return [this]() {
            LOG(INFO) << Indentation() << "Finish applying gradients";
        };
    }

private:

    std::string FormatTensor(const Tensor<T> &tensor) const {
        auto min_max = tensor.template Aggregate<std::optional<std::pair<T, T>>>(
                std::nullopt,
                [](std::optional<std::pair<T, T>> &min_max, const arma::Mat<T> &a) {
                    if (a.n_elem == 0) {
                        return;
                    }
                    if (min_max == std::nullopt) {
                        min_max = std::make_pair(a.min(), a.max());
                    } else {
                        min_max.value().first = std::min(min_max.value().first, a.min());
                        min_max.value().second = std::max(min_max.value().second, a.max());
                    }
                });
        return "[" + FormatDimensions(tensor) +
               ", min value=" + (min_max.has_value() ? std::to_string(min_max.value().first) : "no values") +
               ", max value=" + (min_max.has_value() ? std::to_string(min_max.value().second) : "no values") +
               "]";
    }

    std::string Indentation() {
        return std::string(4 * indentation, ' ');
    }

    int indentation;
    LoggingLevelFlags flags;
};