#pragma once

#include <iomanip>
#include <optional>
#include <iostream>
#include "interface.hpp"
#include "src/neural_network_interface.hpp"
#include "src/tensor.hpp"

template<typename T>
class ProgressBarCallback : public ANeuralNetworkCallback<T> {
public:
    virtual std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *, int epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        return [this](const Tensor<T> &, double loss) {
            loss_history.push_back(loss);
            std::cout << " loss: " << loss << std::endl;
            return CallbackSignal::Continue;
        };
    }

    virtual std::optional<std::function<void()>>
    FitBatch(const nn_framework::data_processing::Data<T> &, int batch_id, int batches_count) {
        std::cerr << "\r" << ProgressBar(batch_id, batches_count) << std::flush;
        return [this, batch_id, batches_count]() {
            std::cerr << "\r" << ProgressBar(batch_id + 1, batches_count) << std::flush;
        };
    }

private:
    std::string ProgressBar(int part, int total) {
        std::stringstream caption_stream;
        caption_stream << part << "/" << total;
        auto caption = std::string(std::max(0, 10 - (int) caption_stream.str().length()), ' ') +
                       caption_stream.str() +
                       " ";
        int percent = 20 * part / total;
        if (part != total) {
            return caption + "[" + std::string(percent, '#') + "*" + std::string(std::max(0, 20 - percent - 1), '.') +
                   "]";
        }
        return caption + "[" + std::string(percent, '#') + "]";
    }

    std::vector<double> loss_history;
};