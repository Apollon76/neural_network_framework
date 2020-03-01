#pragma once

#include "layers.hpp"
#include "optimizer.hpp"
#include <memory>

class NeuralNetwork {
public:
    explicit NeuralNetwork(std::unique_ptr<IOptimizer> _optimizer) : layers(), optimizer(std::move(_optimizer)) {}

    void AddLayer(std::unique_ptr<ILayer> layer) {
        layers.emplace_back(std::move(layer));
    }

    [[nodiscard]] std::string ToString() const {
        std::stringstream output;
        output << "Layers:\n";
        for (auto &layer : layers) {
            output << layer->ToString() << std::endl;
        }
        return output.str();
    }

    void Fit(const arma::mat &input, const arma::mat &output) const {
        DLOG(INFO) << "Fitting neural network...";
        std::vector<arma::mat> inter_outputs = {input};
        for (auto &&layer : layers) {
            DLOG(INFO) << "Fit forward layer: " << layer->GetName();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        std::vector<std::vector<arma::mat>> layer_gradients = {};
        std::vector<arma::mat> output_gradients = {2 * (inter_outputs.back() - output)};
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            DLOG(INFO) << "Propagate gradients backward for layer: " << layers[i]->GetName();
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], output_gradients.back());
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients);
            layers[i]->ApplyGradients(gradients_to_apply);
            output_gradients.push_back(gradients.input_gradients);
        }
        DLOG(INFO) << "Fitting finished";
    }

    [[nodiscard]] arma::mat Predict(const arma::mat &input) const {
        DLOG(INFO) << "Predict with neural network...";
        std::vector<arma::mat> inter_outputs = {input};
        for (auto &&layer : layers) {
            DLOG(INFO) << "Fit forward layer: " << layer->GetName();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        return inter_outputs.back();
    }


private:
    std::vector<std::unique_ptr<ILayer>> layers;
    std::unique_ptr<IOptimizer> optimizer;
};