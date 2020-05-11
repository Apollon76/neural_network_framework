#pragma once

#include <chrono>
#include <iomanip>
#include <src/serialization/hdf5_serialization.hpp>
#include <src/io/filesystem.hpp>
#include "interface.hpp"
#include "src/neural_network_interface.hpp"


template<typename T>
struct EpochMetric {
    int epoch;
    T value;
};

template <typename T>
struct PlottingMetrics {
    Metrics fit_metrics{};
    Metrics fit_batch_metrics{};
    Metrics gradients_metrics{};
    EpochMetric<T> train_score;
    EpochMetric<T> train_loss;
    std::optional<EpochMetric<T>> validation_score = std::nullopt;
    std::optional<EpochMetric<T>> validation_loss = std::nullopt;

    std::map<std::string, Metrics> forward_pass_metrics;
    std::map<std::string, Metrics> backward_pass_metrics;
    std::map<std::string, Metrics> gradient_step_metrics;
    std::map<std::string, Metrics> apply_gradients_metrics;
};

template<typename T>
class PlottingCallback : public ANeuralNetworkCallback<T> {
public:
    PlottingCallback(
        const std::string& logdir,
        const std::string& name,
        std::function<double(const Tensor<T> &, const Tensor<T> &)> &scoring,
        const Tensor<T> &y_train,
        const Tensor<T> &x,
        const Tensor<T> &y,
        int every_nth_epoch = 1
    )
        : logdir(logdir)
        , name(name)
        , scoring(scoring)
        , y_train(y_train)
        , x_val(x)
        , y_val(y)
        , every_nth_epoch(every_nth_epoch)
    {
        EnsureExists();
    }

    virtual std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *nn, int epoch) {
        if (epoch == 0) {
            auto neural_network = dynamic_cast<const NeuralNetwork<T>*>(nn);
            if (neural_network != nullptr) {
                nn_framework::serialization::hdf5::Hdf5Serializer::SaveModel(*neural_network, Path() + "/model.hd5");
            }
        }
        auto start = std::chrono::steady_clock::now();
        return [this, nn, start, epoch](const Tensor<T>& prediction, double loss) {
            metrics.fit_metrics.AddMetric(std::chrono::steady_clock::now() - start);
            metrics.train_score = {epoch, scoring(prediction, y_train)};
            metrics.train_loss = {epoch, loss};
            std::cout << (epoch % every_nth_epoch) << std::endl;
            if (epoch % every_nth_epoch == 0) {
                auto val_prediction = nn->Predict(x_val);
                metrics.validation_score = {epoch, scoring(val_prediction, y_val)};
                metrics.validation_loss = {epoch, nn->GetLoss()->GetLoss(val_prediction, y_val)};
            } else {
                metrics.validation_score = std::nullopt;
                metrics.validation_loss = std::nullopt;
            }
            this->ReportMetrics(epoch);
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
    void EnsureExists() {
        std::experimental::filesystem::remove_all(Path());
        std::experimental::filesystem::create_directories(Path());
        std::experimental::filesystem::create_directories(MetricsPath());
    }

    std::string MetricsPath() {
        return Path() + "/metrics";
    }

    std::string Path() {
        return logdir + "/" + name;
    }

    nlohmann::json getMetricsJson(const Metrics& m, int epoch = 0, bool with_last = false) {
        auto x = nlohmann::json{
                {"average_duration", m.AverageDuration().count()},
        };
        if (with_last) {
            x["epoch"] = epoch;
            x["last_duration"] = m.LastDuration().count();
        }
        return x;
    }

    nlohmann::json getEpochMetricJson(const EpochMetric<T>& m) {
        return nlohmann::json{
            {"epoch", m.epoch},
            {"value", m.value},
        };
    }

    nlohmann::json collectMetricsMap(std::map<std::string, Metrics> &m) {
        nlohmann::json j;
        auto values = std::vector<std::pair<std::string, Metrics>>(m.begin(), m.end());
        std::sort(values.begin(), values.end(),
                  [](const std::pair<std::string, Metrics> &a, const std::pair<std::string, Metrics> &b) {
                      return a.second.AverageDuration() > b.second.AverageDuration();
                  });
        for (auto &&[name, metric] : values) {
            j[name] = getMetricsJson(metric);
        }
        return j;
    }

    void ReportMetrics(int epoch) {
        std::fstream f(MetricsPath() + "/epoch_" + std::to_string(epoch), std::ios_base::out);
        nlohmann::json j = {
                {"fit_metrics", getMetricsJson(metrics.fit_metrics, epoch, true)},
                {"fit_batch_metrics", getMetricsJson(metrics.fit_batch_metrics)},
                {"gradients_metrics", getMetricsJson(metrics.gradients_metrics)},
                {"train_score", getEpochMetricJson(metrics.train_score)},
                {"train_loss", getEpochMetricJson(metrics.train_loss)},
        };

        j["apply_gradients_metrics"] = collectMetricsMap(metrics.apply_gradients_metrics);
        j["backward_pass_metrics"] = collectMetricsMap(metrics.backward_pass_metrics);
        j["forward_pass_metrics"] = collectMetricsMap(metrics.forward_pass_metrics);
        j["gradient_step_metrics"] = collectMetricsMap(metrics.gradient_step_metrics);

        if (metrics.validation_score != std::nullopt) {
            j["validation_score"] = getEpochMetricJson(metrics.validation_score.value());
            j["validation_loss"] = getEpochMetricJson(metrics.validation_loss.value());
        }
        f << j.dump();
        f.close();
    }

private:
    std::string name;
    std::string logdir;
    PlottingMetrics<T> metrics;
    std::function<double(const Tensor<T> &, const Tensor<T> &)> scoring;
    Tensor<T> y_train;
    Tensor<T> x_val;
    Tensor<T> y_val;
    int every_nth_epoch;
};