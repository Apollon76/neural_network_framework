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
    ProgressBarCallback() = default;

    explicit ProgressBarCallback(bool _has_fun) : has_fun(_has_fun) {}

    virtual std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *, int epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        return [this](const Tensor<T> &, double loss) {
            loss_history.push_back(loss);
            std::cout << " loss: " << loss << " " << Fun() << std::endl;
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
    std::string Fun() {
        if (rand() % 4 != 0 || !has_fun) {
            return "";
        }
        if (loss_history.size() == 1) {
            return "good start ಠ‿↼";
        }
        auto a = loss_history[loss_history.size() - 1];
        auto b = loss_history[loss_history.size() - 2];
        if (fabs(a - b) < 1e-4) {
            int option = rand() % 4;
            if (option == 0) {
                return "(¬_¬)";
            } else if (option == 1) {
                return "¯\\_(ツ)_/¯";
            } else if (option == 2) {
                return "add one more dense layer ʘ‿ʘ";
            } else {
                return "something is about to happen...";
            }
        }
        if (a > b) {
            int option = rand() % 4;
            if (option == 0) {
                return "( ⚆ _ ⚆ )";
            } else if (option == 1) {
                return "(~_^)";
            } else if (option == 2) {
                return "(╯°□°)╯︵ ┻━┻";
            } else {
                return std::vector<std::string>{
                        "never gonna give you up...",
                        "never gonna let you down...",
                        "never gonna run around and desert you...",
                        "never gonna make you cry...",
                        "never gonna say goodbye...",
                        "never gonna tell a lie and hurt you...",
                }[rand() % 6];
            }
        }
        int option = rand() % 3;
        if (option == 0) {
            return "♥‿♥";
        } else if (option == 1) {
            return "(~˘▾˘)~ keep going ~(˘▾˘~)";
        } else {
            return "awesome \\ (•◡•) /";
        }
    }

    std::string ProgressBar(int part, int total) {
        auto part_string = std::to_string(part);
        auto total_string = std::to_string(total);
        auto caption =
                std::string(std::max(0, (int) total_string.length() - (int) part_string.length()), ' ') +
                part_string + "/" + total_string + " ";
        int percent = 20 * part / total;
        if (part != total) {
            return caption + "[" + std::string(percent, '=') + ">" + std::string(std::max(0, 20 - percent - 1), '.') +
                   "]";
        }
        return caption + "[" + std::string(percent, '#') + "]";
    }

    std::vector<double> loss_history;
    bool has_fun;
};