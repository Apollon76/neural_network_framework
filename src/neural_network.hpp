#pragma once

#include <memory>
#include <cereal/types/vector.hpp>
#include <pbar.h>
#include <src/callbacks/combined_callback.hpp>

#include "src/layers/activations.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

using nn_framework::data_processing::GenerateBatches;
using nn_framework::data_processing::Data;

static const size_t NoBatches = -1;

template<bool Enabled, typename T, typename Container = std::vector<Data<T>>, size_t Size = 100, char Symbol = '#'>
struct ProgressBar {
    using iterator = typename Container::iterator;
    using value = Container;

    static Container Construct(const Container &container) {
        return Container(container.begin(), container.end());
    }
};

template<typename T, typename Container, size_t Size, char Symbol>
struct ProgressBar<true, T, Container, Size, Symbol> {
    using iterator = typename Container::iterator;
    using value = pbar::ProgressBar<typename Container::iterator>;

    static ProgressBar::value Construct(Container &container) {
        return ProgressBar::value(container.begin(), container.end(), Size, Symbol);
    }
};


template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(
            std::unique_ptr<IOptimizer<T>> _optimizer,
            std::unique_ptr<ILoss<T>> _loss,
            size_t batch_size = NoBatches,
            bool shuffle = false
    ) : layers(),
        callbacks(),
        optimizer(std::move(_optimizer)),
        loss(std::move(_loss)),
        batch_size(batch_size),
        shuffle(shuffle) {
    }

    NeuralNetwork &AddLayer(std::unique_ptr<ILayer<T>> layer) {
        layers.emplace_back(std::move(layer));
        return *this;
    }

    template<template<class> class LayerType, typename... Args>
    NeuralNetwork &AddLayer(Args &&... args) {
        return AddLayer(std::make_unique<LayerType<T>>(std::forward<Args>(args)...));
    }

    NeuralNetwork &AddCallback(std::shared_ptr<ANeuralNetworkCallback<T>> callback) {
        callbacks.emplace_back(std::move(callback));
        return *this;
    }

    template<template<class> class CallbackType, typename... Args>
    NeuralNetwork &AddCallback(Args &&... args) {
        return AddCallback(std::make_shared<CallbackType<T>>(std::forward<Args>(args)...));
    }

    [[nodiscard]] ILayer<T> *GetLayer(size_t layer_id) const {
        return layers[layer_id].get();
    }

    [[nodiscard]] size_t GetLayersCount() const {
        return layers.size();
    }

    [[nodiscard]] IOptimizer<T> *GetOptimizer() const {
        return optimizer.get();
    }

    [[nodiscard]] ILoss<T> *GetLoss() const {
        return loss.get();
    }

    [[nodiscard]] std::string ToString() const {
        std::stringstream output;
        output << "Layers:\n";
        for (auto &layer : layers) {
            output << layer->ToString() << std::endl;
        }
        return output.str();
    }


    void FitNN(int epochs, const Tensor<T> &input, const Tensor<T> &output) {
        auto callback = CombinedNeuralNetworkCallback(callbacks);
        for (int i = 0; i < epochs; i++) {
            auto fitEpochCallback = callback.Fit(i);
            DoFit(input, output, callback);
            if (fitEpochCallback != std::nullopt) {
                auto prediction = Predict(input);
                auto predictionLoss = loss->GetLoss(prediction, output);
                auto signal = fitEpochCallback.value()(prediction, predictionLoss);
                if (signal == CallbackSignal::Stop) {
                    break;
                }
            }
        }
    }

    double Fit(const Tensor<T> &input, const Tensor<T> &output) {
        auto callback = CombinedNeuralNetworkCallback(callbacks);
        DoFit(input, output, callback);
        return loss->GetLoss(Predict(input), output);
    }

    [[nodiscard]] Tensor<T> Predict(const Tensor<T> &input) const {
        Tensor<T> output = input;
        for (auto &&layer : layers) {
            output = layer->Apply(output);
        }
        return output;
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(layers, optimizer, loss);
    }

private:
    void DoFit(const Tensor<T> &input, const Tensor<T> &output, ANeuralNetworkCallback<T> &callback) {
        auto real_batch_size = (batch_size != NoBatches) ? batch_size : input.D[0];

        std::vector<Data<T>> batches = GenerateBatches<T>(Data<T>{input, output}, real_batch_size, shuffle);
        for (const Data<T> &batch: batches) {
            auto fitBatchAction = callback.FitBatch(batch);
            DoFitBatch(batch, callback);
            if (fitBatchAction != std::nullopt) {
                fitBatchAction.value()();
            }
        }
    }

    void DoFitBatch(const Data<T> &batch, ANeuralNetworkCallback<T> &callback) {
        std::vector<Tensor<T>> inter_outputs = {batch.input};
        for (auto &&layer : layers) {
            auto forwardAction = callback.LayerForwardPass(layer.get(), inter_outputs.back());
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
            if (forwardAction != std::nullopt) {
                forwardAction.value()(inter_outputs.back());
            }
        }
//            DLOG(INFO) << "Expected outputs: " << std::endl << batch.output.ToString() << std::endl
//                       << "Actual outputs: " << std::endl << inter_outputs.back().ToString();
        auto gradientsAction = callback.OptimizerGradients(batch.output);
        Tensor<T> last_output_gradient = loss->GetGradients(inter_outputs.back(), batch.output);
        if (gradientsAction != std::nullopt) {
            gradientsAction.value()(last_output_gradient);
        }
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
//                DLOG(INFO) << "Inputs: " << std::endl << inter_outputs[i].ToString() << std::endl
//                           << "Gradients: " << std::endl << last_output_gradient.ToString();
            auto backwardAction = callback.LayerBackwardPass(layers[i].get(), inter_outputs[i], last_output_gradient);
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], last_output_gradient);
            if (backwardAction != std::nullopt) {
                backwardAction.value()(gradients);
            }
//            DLOG(INFO) << "Found gradients: "
//                           << gradients.input_gradients.ToString() << std::endl
//                           << gradients.layer_gradients.ToString() << std::endl;
            auto gradientStepAction = callback.OptimizerGradientStep(layers[i].get(), gradients.layer_gradients);
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients, layers[i].get());
            if (gradientStepAction != std::nullopt) {
                gradientStepAction.value()(gradients_to_apply);
            }
            auto applyGradientsAction = callback.LayerApplyGradients(layers[i].get(), gradients_to_apply);
            layers[i]->ApplyGradients(gradients_to_apply);
            if (applyGradientsAction != std::nullopt) {
                applyGradientsAction.value()();
            }
            last_output_gradient = gradients.input_gradients;
        }
    }

    std::vector<std::unique_ptr<ILayer<T>>> layers;
    std::vector<std::shared_ptr<ANeuralNetworkCallback<T>>> callbacks;
    std::unique_ptr<IOptimizer<T>> optimizer;
    std::unique_ptr<ILoss<T>> loss;
    size_t batch_size;
    bool shuffle;
};
