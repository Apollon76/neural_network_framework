#pragma once

#include <string>

#include "interface.hpp"

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

    std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>> Fit(int epoch) override {
        if (!HasFlag(flags, LoggingLevelFlags::Epoch)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start epoch " << epoch;
        indentation++;
        return [epoch, this](const Tensor<T> &, double loss) {
            indentation--;
            DLOG(INFO) << Indentation() << "Finish epoch " << epoch << " with loss " << loss;
            return CallbackSignal::Continue;
        };
    }

    std::optional<std::function<void()>> FitBatch(const nn_framework::data_processing::Data<T> &batch_data) override {
        if (!HasFlag(flags, LoggingLevelFlags::Batch)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start fitting batch "
                   << "with input tensor " << FormatDimensions(batch_data.input) << " "
                   << "and output tensor " << FormatDimensions(batch_data.output);
        indentation++;
        return [this]() {
            indentation--;
            DLOG(INFO) << Indentation() << "Finish fitting batch";
        };
    }

    std::optional<std::function<void(const Tensor<T> &output)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &input) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start forward pass for layer " << layer->GetName() << " "
                   << "with input tensor " << FormatDimensions(input);
        return [this](const Tensor<T> &) {
            DLOG(INFO) << Indentation() << "Finish forward pass";
        };
    }

    std::optional<std::function<void(const Gradients<T> &gradients)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &input, const Tensor<T> &output_gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation() << "Start backward pass for layer " << layer->GetName()
                   << " with input tensor " << FormatDimensions(input)
                   << " and output gradients tensor " << FormatDimensions(output_gradients);
        return [this](const Gradients<T> &) {
            DLOG(INFO) << Indentation() << "Finish backward pass";
        };
    }

    std::optional<std::function<void(const Tensor<T> &output_gradients)>>
    OptimizerGradients(const Tensor<T> &output) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start calculating output gradients for output tensor " << FormatDimensions(output);
        return [this](const Tensor<T> &) {
            DLOG(INFO) << Indentation() << "Finish calculating output gradients";
        };
    }

    std::optional<std::function<void(const Tensor<T> &gradient_step)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &layer_gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start calculating gradient step for layer " << layer->GetName() << " "
                   << "with layer gradients tensor " << FormatDimensions(layer_gradients);
        return [this](const Tensor<T> &) {
            DLOG(INFO) << Indentation() << "Finish calculating gradient step";
        };
    }

    std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &gradients) override {
        if (!HasFlag(flags, LoggingLevelFlags::Internals)) {
            return std::nullopt;
        }
        DLOG(INFO) << Indentation()
                   << "Start applying gradients for layer " << layer->GetName() << " "
                   << "with layer gradients tensor " << FormatDimensions(gradients);
        return [this]() {
            DLOG(INFO) << Indentation() << "Finish applying gradients";
        };
    }

private:

    std::string Indentation() {
        return std::string(4 * indentation, ' ');
    }

    int indentation;
    LoggingLevelFlags flags;
};