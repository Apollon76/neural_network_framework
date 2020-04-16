#pragma once

#include <memory>
#include <cereal/types/vector.hpp>
#include <pbar.h>

#include "src/layers/activations.hpp"
#include "src/layers/interface.h"
#include "src/data_processing/data_utils.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

using nn_framework::data_processing::GenerateBatches;
using nn_framework::data_processing::Data;

static const size_t NoBatches = -1;

template < bool Enabled, typename T, typename Container = std::vector<Data<T>>, size_t Size = 100, char Symbol = '#'>
struct ProgressBar {
    using iterator = typename Container::iterator;
    using value = Container;

    static Container Construct(const Container& container) {
        return Container(container.begin(), container.end());
    }
};

template <typename T, typename Container, size_t Size, char Symbol>
struct ProgressBar<true, T, Container, Size, Symbol> {
    using iterator = typename Container::iterator;
    using value = pbar::ProgressBar<typename Container::iterator>;

    static ProgressBar::value Construct(Container& container) {
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
        size_t batch_size=NoBatches,
        bool shuffle=false
    )
        : layers()
        , optimizer(std::move(_optimizer))
        , loss(std::move(_loss))
        , batch_size(batch_size)
        , shuffle(shuffle)
    {
    }

    NeuralNetwork &AddLayer(std::unique_ptr<ILayer<T>> layer) {
        layers.emplace_back(std::move(layer));
        return *this;
    }

    template<template<class> class LayerType, typename... Args>
    NeuralNetwork& AddLayer(Args&&... args) {
        return AddLayer(std::make_unique<LayerType<T>>(std::forward<Args>(args)...));
    }

    [[nodiscard]] ILayer<T> *GetLayer(size_t layer_id) const {
        return layers[layer_id].get();
    }

    [[nodiscard]] size_t GetLayersCount() const {
        return layers.size();
    }

    [[nodiscard]] IOptimizer<T>* GetOptimizer() const {
        return optimizer.get();
    }

    [[nodiscard]] ILoss<T>* GetLoss() const {
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

    template <bool BarEnabled=false, typename ProgressBarT=ProgressBar<BarEnabled, T>>
    double Fit(const Tensor<T> &input, const Tensor<T> &output) {
        DLOG(INFO) << "Fitting neural network...";

        auto real_batch_size = (batch_size != NoBatches) ? batch_size : input.D[0];

        std::vector<Data<T>> batches = GenerateBatches<T>(Data<T>{input, output}, real_batch_size, shuffle);
        auto bar = ProgressBarT::Construct(batches);

        for (const Data<T>& batch: bar) {
            std::vector<Tensor<T>> inter_outputs = {batch.input};
            DLOG(INFO) << "Batch: [input=" << FormatDimensions(batch.input.D)
                       << " output=" << FormatDimensions(batch.output.D) << "]";
            for (auto &&layer : layers) {
                DLOG(INFO) << "Fit forward layer: " << layer->GetName();
                DLOG(INFO) << "Input dim: " << FormatDimensions(inter_outputs.back().D);
                inter_outputs.push_back(layer->Apply(inter_outputs.back()));
                DLOG(INFO) << "Output dim: " << FormatDimensions(inter_outputs.back().D);
            }
//            DLOG(INFO) << "Expected outputs: " << std::endl << batch.output.ToString() << std::endl
//                       << "Actual outputs: " << std::endl << inter_outputs.back().ToString();
            Tensor<T> last_output_gradient = loss->GetGradients(inter_outputs.back(), batch.output);
            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
                DLOG(INFO) << "Propagate gradients backward for layer: " << layers[i]->GetName();
//                DLOG(INFO) << "Inputs: " << std::endl << inter_outputs[i].ToString() << std::endl
//                           << "Gradients: " << std::endl << last_output_gradient.ToString();
                auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], last_output_gradient);
//                DLOG(INFO) << "Found gradients: "
//                           << gradients.input_gradients.ToString() << std::endl
//                           << gradients.layer_gradients.ToString() << std::endl;
                auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients, layers[i].get());
                DLOG(INFO) << "Optimizer applied...";
                layers[i]->ApplyGradients(gradients_to_apply);
                DLOG(INFO) << "Gradients applied...";
                last_output_gradient = gradients.input_gradients;
                DLOG(INFO) << "Last output gradient updated...";
            }
        }

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
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    std::unique_ptr<IOptimizer<T>> optimizer;
    std::unique_ptr<ILoss<T>> loss;
    size_t batch_size;
    bool shuffle;
};
