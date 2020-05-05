#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <functional>

#include "src/callbacks/interface.hpp"
#include "src/tensor.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"


// todo (sivukhin): add simple error handling in CombinedNeuralNetworkCallback
template<typename T>
class CombinedNeuralNetworkCallback : public ANeuralNetworkCallback<T> {
public:
    explicit CombinedNeuralNetworkCallback(std::vector<std::shared_ptr<ANeuralNetworkCallback<T>>> _callbacks)
            : callbacks(_callbacks) {}

    std::optional<std::function<CallbackSignal(const Tensor<T> &, double)>> Fit(int epoch) override {
        std::optional<std::function<CallbackSignal(const Tensor<T> &, double)>> action = std::nullopt;
        for (auto &&callback : callbacks) {
            auto current = callback->Fit(epoch);
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

    std::optional<std::function<void()>> FitBatch(const nn_framework::data_processing::Data<T> &batch_data) override {
        auto finishFuncs = FeedCallbacks([&batch_data](ANeuralNetworkCallback<T> &c) {
            return c.FitBatch(batch_data);
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
