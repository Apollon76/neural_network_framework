#pragma once

#include <chrono>
#include "interface.hpp"

class Metrics {
public:
    [[nodiscard]] std::chrono::milliseconds AverageDuration() const {
        return total_duration / metrics_count;
    }

    [[nodiscard]] std::chrono::milliseconds LastDuration() const {
        return last_duration;
    }

    [[nodiscard]] int64_t MetricsCount() const {
        return metrics_count;
    }

    template<typename Duration>
    void AddMetric(const Duration &duration) {
        last_duration = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        total_duration += std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        metrics_count++;
    }

private:
    std::chrono::milliseconds total_duration;
    std::chrono::milliseconds last_duration;
    int64_t metrics_count;
};

struct PerformanceMetrics {
    Metrics fit_metrics;
    Metrics fit_batch_metrics;
    Metrics gradients_metrics;
    std::map<std::string, Metrics> forward_pass_metrics;
    std::map<std::string, Metrics> backward_pass_metrics;
    std::map<std::string, Metrics> gradient_step_metrics;
    std::map<std::string, Metrics> apply_gradients_metrics;
};

template<typename T>
class PerformanceMetricsCallback : public ANeuralNetworkCallback<T> {
public:
    PerformanceMetricsCallback() = default;

    std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>> Fit(int) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start](const Tensor<T> &, double) {
            metrics.fit_metrics.AddMetric(std::chrono::steady_clock::now() - start);
            return CallbackSignal::Continue;
        };
    }

    std::optional<std::function<void()>> FitBatch(const nn_framework::data_processing::Data<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start]() {
            metrics.fit_batch_metrics.AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &output)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start, layer](const Tensor<T> &) {
            metrics.forward_pass_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Gradients<T> &gradients)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start, layer](const Gradients<T> &) {
            metrics.backward_pass_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &output_gradients)>>
    OptimizerGradients(const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start](const Tensor<T> &) {
            metrics.gradients_metrics.AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &gradient_step)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start, layer](const Tensor<T> &) {
            metrics.gradient_step_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, &start, layer]() {
            metrics.apply_gradients_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

private:
    PerformanceMetrics metrics;
};