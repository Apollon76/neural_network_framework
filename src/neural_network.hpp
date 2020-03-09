#pragma once

#include "layers.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include <memory>

class NeuralNetwork {
public:
    explicit NeuralNetwork(std::unique_ptr<IOptimizer> _optimizer, std::unique_ptr<ILoss> _loss)
            : layers(), optimizer(std::move(_optimizer)), loss(std::move(_loss)) {

    }

    NeuralNetwork& AddLayer(std::unique_ptr<ILayer> layer) {
        layers.emplace_back(std::move(layer));
        return *this;
    }

    [[nodiscard]] ILayer *GetLayer(int layer_id) const {
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

    double Fit(const arma::mat &input, const arma::mat &output) {
        DLOG(INFO) << "Fitting neural network...";
        std::vector<arma::mat> inter_outputs = {input};
        for (auto &&layer : layers) {
            DLOG(INFO) << "Fit forward layer: " << layer->GetName();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        std::vector<std::vector<arma::mat>> layer_gradients = {};
        DLOG(INFO) << "Expected outputs: " << std::endl << output << std::endl
                   << "Actual outputs: " << std::endl << inter_outputs.back();
        std::vector<arma::mat> output_gradients = {loss->GetGradients(inter_outputs.back(), output)};
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            DLOG(INFO) << "Propagate gradients backward for layer: " << layers[i]->GetName();
            DLOG(INFO) << "Intermediate output: " << std::endl << inter_outputs[i] << std::endl
                       << "Output gradients: " << std::endl << output_gradients.back();
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], output_gradients.back());
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients, layers[i].get());
            DLOG(INFO) << "Gradients for layer: " << layers[i]->GetName() << std::endl
                       << gradients.layer_gradients;
            layers[i]->ApplyGradients(gradients_to_apply);
            output_gradients.push_back(gradients.input_gradients);
        }
        DLOG(INFO) << "Fitting finished";
        return loss->GetLoss(inter_outputs.back(), output);
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
    std::unique_ptr<ILoss> loss;
};