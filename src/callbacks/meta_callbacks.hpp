#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <functional>

#include "src/callbacks/interface.hpp"
#include "src/tensor.hpp"
#include "src/neural_network_interface.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"
#include "src/loss.hpp"


// todo (sivukhin): add simple error handling in CombinedNeuralNetworkCallback
template<typename T>
class CombinedNeuralNetworkCallback : public ANeuralNetworkCallback<T> {
public:
    explicit CombinedNeuralNetworkCallback() {}

    explicit CombinedNeuralNetworkCallback(std::vector<std::shared_ptr<ANeuralNetworkCallback<T>>> _callbacks)
            : callbacks(_callbacks) {}

    std::optional<std::function<CallbackSignal(const Tensor<T> &, double)>>
    Fit(const INeuralNetwork<T> *nn, int epoch) override {
        std::optional<std::function<CallbackSignal(const Tensor<T> &, double)>> action = std::nullopt;
        for (auto &&callback : callbacks) {
            auto current = callback->Fit(nn, epoch);
            if (current != std::nullopt) {
                action = [action, current](const Tensor<T> &prediction, double loss) {
                    auto signal = current.value()(prediction, loss);
                    auto otherSignals = action == std::nullopt ?
                                        CallbackSignal::Continue :
                                        action.value()(prediction, loss);
                    return signal == CallbackSignal::Stop || otherSignals == CallbackSignal::Stop
                           ? CallbackSignal::Stop : CallbackSignal::Continue;
                };
            }
        }
        return action;
    }

    std::optional<std::function<void()>>
    FitBatch(const nn_framework::data_processing::Data<T> &batch_data, int batch_id, int batches_count) override {
        auto finishFuncs = FeedCallbacks([&batch_data, batch_id, batches_count](ANeuralNetworkCallback<T> &c) {
            return c.FitBatch(batch_data, batch_id, batches_count);
        });
        return [finishFuncs, this]() {
            FinishCallbacks(finishFuncs);
        };
    }

    std::optional<std::function<void(const Tensor<T> &)>>
    LayerForwardPass(const ILayer<T> *layer, const Tensor<T> &input) override {
        auto finishFuncs = FeedCallbacks([layer, &input](ANeuralNetworkCallback<T> &c) {
            return c.LayerForwardPass(layer, input);
        });
        return [finishFuncs, this](const Tensor<T> &output) {
            FinishCallbacks(finishFuncs, output);
        };
    }

    std::optional<std::function<void(const Gradients<T> &)>>
    LayerBackwardPass(const ILayer<T> *layer, const Tensor<T> &input, const Tensor<T> &output_gradients) override {
        auto finishFuncs = FeedCallbacks([layer, &input, &output_gradients](ANeuralNetworkCallback<T> &c) {
            return c.LayerBackwardPass(layer, input, output_gradients);
        });
        return [finishFuncs, this](const Gradients<T> &gradients) {
            FinishCallbacks(finishFuncs, gradients);
        };
    }

    std::optional<std::function<void(const Tensor<T> &)>> OptimizerGradients(const Tensor<T> &output) override {
        auto finishFuncs = FeedCallbacks([&output](ANeuralNetworkCallback<T> &c) {
            return c.OptimizerGradients(output);
        });
        return [finishFuncs, this](const Tensor<T> &output_gradients) {
            FinishCallbacks(finishFuncs, output_gradients);
        };
    }

    std::optional<std::function<void(const Tensor<T> &)>>
    OptimizerGradientStep(const ILayer<T> *layer, const Tensor<T> &layer_gradients) override {
        auto finishFuncs = FeedCallbacks([layer, &layer_gradients](ANeuralNetworkCallback<T> &c) {
            return c.OptimizerGradientStep(layer, layer_gradients);
        });
        return [finishFuncs, this](const Tensor<T> &gradient_step) {
            FinishCallbacks(finishFuncs, gradient_step);
        };
    }

    std::optional<std::function<void()>>
    LayerApplyGradients(const ILayer<T> *layer, const Tensor<T> &gradients) override {
        auto finishFuncs = FeedCallbacks([layer, &gradients](ANeuralNetworkCallback<T> &c) {
            return c.LayerApplyGradients(layer, gradients);
        });
        return [finishFuncs, this]() {
            FinishCallbacks(finishFuncs);
        };
    }

private:
    template<
            typename Func,
            typename Output = decltype(std::declval<Func>()(std::declval<ANeuralNetworkCallback<T> &>()))
    >
    std::vector<Output>
    FeedCallbacks(Func callbackTrigger) {
        auto funcs = std::vector<Output>();
        funcs.reserve(callbacks.size());
        for (auto &&callback : callbacks) {
            funcs.push_back(callbackTrigger(*callback));
        }
        return funcs;
    }

    template<typename Func, typename... Args>
    void FinishCallbacks(const std::vector<std::optional<Func>> &funcs, Args &&... args) {
        for (int i = (int) funcs.size() - 1; i >= 0; i--) {
            auto func = funcs[i];
            if (func != std::nullopt) {
                func.value()(std::forward<Args>(args)...);
            }
        }
    }

    std::vector<std::shared_ptr<ANeuralNetworkCallback<T>>> callbacks;
};

template<typename T>
using FitCallback = std::optional<std::function<CallbackSignal(const Tensor<T> &, double)>>;

template<typename T>
class EpochCallback : public ANeuralNetworkCallback<T> {
public:
    explicit EpochCallback(const std::function<FitCallback<T>(const INeuralNetwork<T> *, int)> &_callback)
            : callback(_callback) {}


    std::optional<std::function<CallbackSignal(const Tensor<T> &prediction, double loss)>>
    Fit(const INeuralNetwork<T> *nn, int epoch) override {
        return callback(nn, epoch);
    }

private:
    std::function<FitCallback<T>(const INeuralNetwork<T> *, int)> callback;
};

template<typename T>
std::shared_ptr<ANeuralNetworkCallback<T>>
EveryNthEpoch(int every_nth_epoch, std::shared_ptr<ANeuralNetworkCallback<T>> callback) {
    return std::make_shared<EpochCallback<T>>(
            [every_nth_epoch, callback](const INeuralNetwork<T> *nn, int epoch) -> FitCallback<T> {
                if (epoch % every_nth_epoch == 0) {
                    return callback->Fit(nn, epoch);
                }
                return std::nullopt;
            }
    );
}

template<typename T>
std::shared_ptr<ANeuralNetworkCallback<T>>
ScoreCallback(const std::string &caption,
              const std::function<double(const Tensor<T> &, const Tensor<T> &)> &scoring,
              const Tensor<T> &x, const Tensor<T> &y) {
    return std::make_shared<EpochCallback<T>>(
            [&x, &y, scoring, caption](const INeuralNetwork<T> *nn, int epoch) -> FitCallback<T> {
                return [&x, &y, scoring, caption, nn, epoch](const Tensor<T> &, double) {
                    auto prediction = nn->Predict(x);
                    auto l = scoring(prediction, y);
                    LOG(INFO) << "Epoch " << epoch << ", " << caption << ": " << l;
                    return CallbackSignal::Continue;
                };
            });
}
