#pragma once

#include <memory>
#include <cereal/types/vector.hpp>
#include <pbar.h>
#include <src/callbacks/meta_callbacks.hpp>

#include "src/layers/activations.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "neural_network_interface.hpp"
#include "initializers.hpp"

using nn_framework::data_processing::GenerateBatches;
using nn_framework::data_processing::Data;

static const size_t NoBatches = -1;

template<typename T>
class NeuralNetwork : public INeuralNetwork<T> {
public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(
            std::unique_ptr<IOptimizer<T>> _optimizer,
            std::unique_ptr<ILoss<T>> _loss
    ) : layers(),
        optimizer(std::move(_optimizer)),
        loss(std::move(_loss)) {
        initializer = std::make_unique<UniformInitializer>();
    }

    explicit NeuralNetwork(
            std::unique_ptr<IOptimizer<T>> _optimizer,
            std::unique_ptr<ILoss<T>> _loss,
            std::unique_ptr<IInitializer> _initializer
    ) : layers(),
        optimizer(std::move(_optimizer)),
        loss(std::move(_loss)),
        initializer(std::move(_initializer)) {
    }

    NeuralNetwork &AddLayer(std::unique_ptr<ILayer<T>> layer) {
        layer->Initialize(initializer);
        layers.emplace_back(std::move(layer));
        return *this;
    }

    template<template<class> class LayerType, typename... Args>
    NeuralNetwork &AddLayer(Args &&... args) {
        return AddLayer(std::make_unique<LayerType<T>>(std::forward<Args>(args)...));
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

    [[nodiscard]] ILoss<T> *GetLoss() const override {
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

    void Fit(const Tensor<T> &input, const Tensor<T> &output, int epochs, size_t batch_size = NoBatches,
             bool shuffle = false, std::shared_ptr<ANeuralNetworkCallback<T>> callback = nullptr) {
        if (callback == nullptr) {
            callback = std::make_shared<CombinedNeuralNetworkCallback<T>>();
        }
        for (int i = 0; i < epochs; i++) {
            auto fitEpochCallback = callback->Fit(this, i);
            DoFit(input, output, batch_size, shuffle, *callback);
            if (fitEpochCallback != std::nullopt) {
                // todo (sivukhin): try to avoid unnecessary calculation here
                auto prediction = Predict(input);
                auto predictionLoss = loss->GetLoss(prediction, output);
                auto signal = fitEpochCallback.value()(prediction, predictionLoss);
                if (signal == CallbackSignal::Stop) {
                    break;
                }
            }
        }
    }

    double FitOneIteration(const Tensor<T> &input, const Tensor<T> &output) {
        auto callback = CombinedNeuralNetworkCallback<T>();
        DoFit(input, output, NoBatches, false, callback);
        return loss->GetLoss(Predict(input), output);
    }

    [[nodiscard]] Tensor<T> Predict(const Tensor<T> &input) const override {
        Tensor<T> output = input;
        for (auto &&layer : layers) {
            output = layer->Apply(output);
        }
        return output;
    }

    void SetTrain(bool value) override {
        for (auto& layer : layers) {
            layer->SetTrain(value);
        }
    }

    template<class Archive>
    void serialize(Archive &ar) {
        ar(layers, optimizer, loss);
    }

private:
    void DoFit(const Tensor<T> &input, const Tensor<T> &output, size_t batch_size, bool shuffle,
               ANeuralNetworkCallback<T> &callback) {
        auto real_batch_size = (batch_size != NoBatches) ? batch_size : input.D[0];

        std::vector<Data<T>> batches = GenerateBatches<T>(Data<T>{input, output}, real_batch_size, shuffle);
        int batch_id = 0;
        for (const Data<T> &batch: batches) {
            auto fitBatchAction = callback.FitBatch(batch, batch_id++, batches.size());
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
        auto gradientsAction = callback.OptimizerGradients(batch.output);
        Tensor<T> last_output_gradient = loss->GetGradients(inter_outputs.back(), batch.output);
        if (gradientsAction != std::nullopt) {
            gradientsAction.value()(last_output_gradient);
        }
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            auto backwardAction = callback.LayerBackwardPass(layers[i].get(), inter_outputs[i], last_output_gradient);
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], last_output_gradient);
            if (backwardAction != std::nullopt) {
                backwardAction.value()(gradients);
            }
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
    std::unique_ptr<IOptimizer<T>> optimizer;
    std::unique_ptr<ILoss<T>> loss;
    std::unique_ptr<IInitializer> initializer;
};
