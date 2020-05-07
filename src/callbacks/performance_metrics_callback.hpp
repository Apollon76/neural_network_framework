#pragma once

#include <chrono>
#include <iomanip>
#include "interface.hpp"
#include "src/neural_network_interface.hpp"

class Metrics {
public:
    [[nodiscard]] std::chrono::milliseconds AverageDuration() const {
        if (metrics_count == 0) {
            return std::chrono::milliseconds::zero();
        }
        return total_duration / metrics_count;
    }

    [[nodiscard]] std::chrono::milliseconds LastDuration() const {
        return last_duration;
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

std::string ReportMetric(const std::string &caption, Metrics &metrics, bool with_last_duration) {
    std::stringstream report;
    report << caption << ": average duration=" << metrics.AverageDuration().count() << "ms";
    if (with_last_duration) {
        report << ", last duration=" << metrics.LastDuration().count() << "ms";
    }
    report << std::endl;
    return report.str();
}

std::string ReportMetricMap(const std::string &caption, std::map<std::string, Metrics> &metrics) {
    std::stringstream report;
    report << caption << std::endl;
    auto values = std::vector<std::pair<std::string, Metrics>>(metrics.begin(), metrics.end());
    std::sort(values.begin(), values.end(),
              [](const std::pair<std::string, Metrics> &a, const std::pair<std::string, Metrics> &b) {
                  return a.second.AverageDuration() > b.second.AverageDuration();
              });
    for (auto &&[name, metric] : values) {
        report << std::string(8, ' ') << std::setw(8) << metric.AverageDuration().count();
        report << "ms: " << name << std::endl;
    }
    return report.str();
}

std::string ReportMetrics(PerformanceMetrics &full_metrics) {
    auto indent = std::string(4, ' ');
    std::stringstream report;
    report << "Metrics report: " << std::endl;
    report << indent << ReportMetric("Full epoch          ", full_metrics.fit_metrics, true);
    report << indent << ReportMetric("Full batch          ", full_metrics.fit_batch_metrics, false);
    report << indent << ReportMetric("Gradient calculation", full_metrics.gradients_metrics, false);
    report << indent << ReportMetricMap("Forward pass", full_metrics.forward_pass_metrics);
    report << indent << ReportMetricMap("Backward pass", full_metrics.backward_pass_metrics);
    report << indent << ReportMetricMap("Gradient step", full_metrics.gradient_step_metrics);
    report << indent << ReportMetricMap("Apply gradients", full_metrics.apply_gradients_metrics);
    return report.str();
}

template<typename T>
class PerformanceMetricsCallback : public ANeuralNetworkCallback<T> {
public:
    PerformanceMetricsCallback() = default;

    std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *, int) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start](const Tensor<T> &, double) {
            metrics.fit_metrics.AddMetric(std::chrono::steady_clock::now() - start);
            std::cout << ReportMetrics(metrics);
            return CallbackSignal::Continue;
        };
    }

    std::optional<std::function<void()>> FitBatch(const nn_framework::data_processing::Data<T> &, int, int) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start]() {
            metrics.fit_batch_metrics.AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &output)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start, layer](const Tensor<T> &) {
            metrics.forward_pass_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Gradients<T> &gradients)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start, layer](const Gradients<T> &) {
            metrics.backward_pass_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &output_gradients)>>
    OptimizerGradients(const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start](const Tensor<T> &) {
            metrics.gradients_metrics.AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void(const Tensor<T> &gradient_step)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start, layer](const Tensor<T> &) {
            metrics.gradient_step_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

    std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &) override {
        auto start = std::chrono::steady_clock::now();
        return [this, start, layer]() {
            metrics.apply_gradients_metrics[layer->GetName()].AddMetric(std::chrono::steady_clock::now() - start);
        };
    }

private:
    PerformanceMetrics metrics;
};