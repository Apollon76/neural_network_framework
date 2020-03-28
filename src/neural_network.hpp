#pragma once

#include <memory>
#include <cereal/types/vector.hpp>

#include <src/layers/activations.hpp>
#include <src/layers/interface.h>

#include "loss.hpp"
#include "optimizer.hpp"
#include <src/tensor.hpp>

template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(std::unique_ptr<IOptimizer<T>> _optimizer, std::unique_ptr<ILoss<T>> _loss)
            : layers(), optimizer(std::move(_optimizer)), loss(std::move(_loss)) {

    }

    NeuralNetwork &AddLayer(std::unique_ptr<ILayer<T>> layer) {
        layers.emplace_back(std::move(layer));
        return *this;
    }

    [[nodiscard]] ILayer<T> *GetLayer(int layer_id) const {
        return layers[layer_id].get();
    }

    [[nodiscard]] std::string ToString() const {
        std::stringstream output;
        output << "Layers:\n";
        for (auto &layer : layers) {
            output << layer->ToString() << std::endl;
        }
        return output.str();
    }

    double Fit(const Tensor<T> &input, const Tensor<T> &output) {
        std::vector<Tensor<T>> inter_outputs = {input};
        for (auto &&layer : layers) {
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        Tensor<T> last_output_gradient = loss->GetGradients(inter_outputs.back(), output);
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], last_output_gradient);
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients, layers[i].get());
            layers[i]->ApplyGradients(gradients_to_apply);
            last_output_gradient = gradients.input_gradients;
        }
        return loss->GetLoss(inter_outputs.back(), output);
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
};
