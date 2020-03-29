#pragma once

#include <string>
#include <memory>
#include <cereal/types/vector.hpp>

#include <src/layers/activations.hpp>
#include <src/layers/interface.h>

#include "loss.hpp"
#include "optimizer.hpp"
#include <src/tensor.hpp>
#include "layers/layers_enum.hpp"
#include <layers.pb.h>

class InputStream : public google::protobuf::io::CopyingInputStream, std::istream {
private:
    std::istream *innerStream;
public:
    InputStream(std::istream *innerStream): innerStream(innerStream) {}

    int Read(void* buffer, int size) override {
        return innerStream->readsome(reinterpret_cast<char*>(buffer), size);
    }
};


class OutputStream : public google::protobuf::io::CopyingOutputStream, std::ostream {
private:
    std::ostream *innerStream;
public:
    OutputStream(std::ostream* innerStream): innerStream(innerStream) {}

    bool Write(const void* buffer, int size) override {
        innerStream->write(reinterpret_cast<const char*>(buffer), size);
        return !(innerStream->rdstate() && (innerStream->failbit || innerStream->badbit));
    }

};


template<typename T>
class NeuralNetwork {
public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(std::unique_ptr<IOptimizer<T>> _optimizer, std::unique_ptr<ILoss<T>> _loss)
            : layers(), optimizer(std::move(_optimizer)), loss(std::move(_loss)), layerByTypeCounter(LayersEnum::TOTAL) {

    }

    NeuralNetwork &AddLayer(std::unique_ptr<ILayer<T>> layer) {
        LayersEnum layerType = layer->GetLayerType();
        std::string layerName = GetLayerNameByType(layerType);

        size_t vacantID = layerByTypeCounter[layerType]++;
        layer->SetLayerID(layerName + "_" + std::to_string(vacantID));
        layerId2layer[layer->GetLayerID()] = layers.size();

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
        DLOG(INFO) << "Fitting neural network...";
        std::vector<Tensor<T>> inter_outputs = {input};
        for (auto &&layer : layers) {
            DLOG(INFO) << "Fit forward layer: " << layer->GetName();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        DLOG(INFO) << "Expected outputs: " << std::endl << output.ToString() << std::endl
                   << "Actual outputs: " << std::endl << inter_outputs.back().ToString();
        Tensor<T> last_output_gradient = loss->GetGradients(inter_outputs.back(), output);
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            DLOG(INFO) << "Propagate gradients backward for layer: " << layers[i]->GetName();
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], last_output_gradient);
            DLOG(INFO) << "Found gradients: "
                       << gradients.input_gradients.ToString() << std::endl
                       << gradients.layer_gradients.ToString() << std::endl;
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients, layers[i].get());
            DLOG(INFO) << "Optimizer applied...";
            layers[i]->ApplyGradients(gradients_to_apply);
            DLOG(INFO) << "Gradients applied...";
            last_output_gradient = gradients.input_gradients;
            DLOG(INFO) << "Last output gradient updated...";
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

    void SaveWeights(google::protobuf::io::CopyingOutputStreamAdaptor *out) const {
        google::protobuf::io::CodedOutputStream output(out);
        for (auto && layer : layers) {
            if (layer->GetLayerType() == LayersEnum::DENSE) {
                std::stringstream layer_weights;
                size_t layerSize = layer->SaveWeights(&layer_weights);
                output.WriteVarint32(layer->GetLayerType());
                output.WriteVarint32(layerSize);
                output.WriteRaw(layer_weights.str().c_str(), layerSize);
            }
        }
    }

    void LoadWeights(google::protobuf::io::CopyingInputStreamAdaptor *rawInput) {
        google::protobuf::io::CodedInputStream input(rawInput);

        while (!input.ConsumedEntireMessage()) {
            uint32_t layerTypeInt;
            input.ReadVarint32(&layerTypeInt);
            auto layerType = LayersEnum(layerTypeInt);

            uint32_t size;
            input.ReadVarint32(&size);
            google::protobuf::io::CodedInputStream::Limit limit = input.PushLimit(size);

            if (layerType == LayersEnum::DENSE) {
                DenseWeights dense;
                dense.MergeFromCodedStream(&input);
                if (auto it = layerId2layer.find(dense.name()); it != layerId2layer.end()) {
                    (DenseLayer<T>(*it->second)).LoadWeights(dense);
                }
            }
            input.PopLimit(limit);
        }
    }

private:
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    std::unique_ptr<IOptimizer<T>> optimizer;
    std::unique_ptr<ILoss<T>> loss;
    std::vector<size_t> layerByTypeCounter;
    std::unordered_map<std::string, size_t> layerId2layer;
};
