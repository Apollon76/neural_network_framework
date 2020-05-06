#pragma once

#include <src/neural_network.hpp>
#include <src/layers/dense.hpp>

#include <highfive/H5Easy.hpp>
#include <nlohmann/json.hpp>
#include <hdf5/serial/H5Apublic.h>
#include <hdf5/serial/H5Ppublic.h>

namespace nn_framework::serialization::hdf5 {
    class Hdf5Serializer {
    public:
        static NeuralNetwork<double> LoadModel(const std::string& filename) {
            HighFive::File file(filename, HighFive::File::ReadOnly);

            auto model_config_attr = file.getAttribute("model_config");
            auto training_config_attr = file.getAttribute("training_config");

            auto model_config = nlohmann::json::parse(ReadAttribute(model_config_attr));
            auto training_config = nlohmann::json::parse(ReadAttribute(training_config_attr));

            auto optimizer = load_optimizer<double>(training_config);
            auto loss = get_loss<double>(training_config);

            auto model = NeuralNetwork<double>(std::move(optimizer), std::move(loss));

            auto mapping = load_layers(model_config, &model);

            load_weights(file.getGroup("model_weights"), mapping, &model);

            return model;
        }

        template<class T>
        static void SaveModel(const NeuralNetwork<T>& model, const std::string& filename) {
            HighFive::File file(filename, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

            auto mapping = save_weights(model, file.createGroup("model_weights"));

            auto model_config = save_layers(model, mapping);
            WriteAttribute(file, "model_config", model_config.dump());

            auto training_config = save_training_config(model);
            WriteAttribute(file, "training_config", training_config.dump());
        }

    private:
        static std::string ReadAttribute(const HighFive::Attribute& attribute) {
            char* buf;
            H5Aread(attribute.getId(), attribute.getDataType().getId(), &buf);
            return std::string(buf);
        }

        static void WriteAttribute(const HighFive::Object& object, const std::string& name, const std::string& value) {
            auto space_hid = H5Screate_simple(0, nullptr, nullptr);
            ensure(space_hid > 0);

            auto type = H5Tcopy(H5T_C_S1);
            H5Tset_size(type, value.length());

            auto attr_hid = H5Acreate2(object.getId(), name.c_str(), type, space_hid, H5P_DEFAULT, H5P_DEFAULT);
            ensure(attr_hid > 0);

            H5Awrite(attr_hid, type, value.data());
        }

        static void WriteAttribute(const HighFive::Object &object, const std::string& name,
                                   const std::vector<std::string>& value) {
            std::vector<hsize_t> dims = {value.size()};
            auto space_hid = H5Screate_simple(1, dims.data(), nullptr);
            ensure(space_hid > 0);

            auto type = H5Tcopy(H5T_C_S1);
            H5Tset_size(type, H5T_VARIABLE);

            auto attr_hid = H5Acreate2(object.getId(), name.c_str(), type, space_hid, H5P_DEFAULT, H5P_DEFAULT);
            ensure(attr_hid > 0);

            std::vector<const char*> charValues;
            charValues.reserve(value.size());
            for (const auto& v : value) {
                charValues.push_back(v.c_str());
            }

            H5Awrite(attr_hid, type, charValues.data());
        }

        template <class T>
        static std::unique_ptr<IOptimizer<T>> load_optimizer(const nlohmann::json& training_config) {
            const auto& optimizer_config = training_config["optimizer_config"]["config"];
            auto optimizer_name = optimizer_config["name"].get<std::string>();
            if (optimizer_name == "Adam") {
                auto learning_rate = optimizer_config["learning_rate"].get<double>();
                //auto decay = optimizer_config["decay"].get<double>();
                auto beta_1 = optimizer_config["beta_1"].get<double>();
                auto beta_2 = optimizer_config["beta_2"].get<double>();
                auto epsilon = optimizer_config["epsilon"].get<double>();
                //auto amsgrad = optimizer_config["amsgrad"].get<bool>();
                return std::make_unique<AdamOptimizer<T>>(learning_rate, beta_1, beta_2, epsilon);
            } else {
                throw std::runtime_error("Unknown optimizer: " + optimizer_name);
            }
        }

        template <class T>
        static std::unique_ptr<ILoss<T>> get_loss(const nlohmann::json& training_config) {
            auto loss_name = training_config["loss"]["class_name"].get<std::string>();
            if (loss_name == "CategoricalCrossentropy") {
                return std::make_unique<CategoricalCrossEntropyLoss<T>>();
            } else {
                throw std::runtime_error("Unknown loss: " + loss_name);
            }
        }

        template <class T>
        static nlohmann::json save_training_config(const NeuralNetwork<T>& model) {
            nlohmann::json result = {
                    {"metrics", {"accuracy"}}, // todo
                    {"weighted_metrics", nullptr},
                    {"sample_weight_mode", nullptr},
                    {"loss_weights", nullptr},
            };

            auto loss = model.GetLoss();
            auto optimizer = model.GetOptimizer();

            if (auto categorical_cross_entropy = dynamic_cast<CategoricalCrossEntropyLoss<T>*>(loss)) {
                result["loss"] = {
                    {"class_name", "CategoricalCrossentropy"},
                    {"config", {
                            {"reduction", "auto"},
                            {"name", "categorical_crossentropy"},
                            {"from_logits", true}, // todo
                            {"label_smoothing", 0}
                        }
                    }
                };
            } else {
                throw std::runtime_error("Unsupported loss");
            }

            if (auto adam = dynamic_cast<AdamOptimizer<T>*>(optimizer)) {
                result["optimizer_config"] = {
                    {"class_name", "Adam"},
                    {"config", {
                            {"name", "Adam"},
                            {"learning_rate", adam->getLearningRate()},
                            {"decay", 0.0}, // todo
                            {"beta_1", adam->getBeta1()},
                            {"beta_2", adam->getBeta2()},
                            {"epsilon", adam->getEpsilon()},
                            {"amsgrad", false} // todo
                        }
                    }
                };
            } else {
                throw std::runtime_error("Unsupported optimizer");
            }

            return result;
        }

        template <class T>
        static std::unordered_map<size_t, std::string> load_layers(const nlohmann::json& model_config, NeuralNetwork<T>* model) {
            ensure(model_config["class_name"].get<std::string>() == "Sequential", "Only Sequential models are supported");
            auto layers = model_config["config"]["layers"];
            std::vector<size_t> last_layer_output_dims;
            std::unordered_map<size_t, std::string> mapping;
            size_t addedLayerIndex = 0;
            for (size_t i = 0; i < layers.size(); i++) {
                const auto& layer_item = layers[i];
                auto layer_name = layer_item["class_name"].get<std::string>();
                auto layer_config = layer_item["config"];
                if (layer_name == "Dense") {
                    ensure(layer_config["trainable"].get<bool>(), "Non-trainable layers are not supported");

                    std::vector<size_t> dims;
                    if (i == 0) {
                        ensure(layer_config.contains("batch_input_shape"), "First layer should contain batch_input_shape");
                        for (const auto &dim: layer_config["batch_input_shape"]) {
                            if (!dim.is_null()) {
                                dims.push_back(dim.get<size_t>());
                            }
                        }
                    } else {
                        dims = last_layer_output_dims;
                    }

                    ensure(dims.size() == 1, "Input shape of dense layer should be one-dimensional (got " +
                                             std::to_string(dims.size()) + " dimensions)");
                    auto input_size = dims[0];
                    auto output_size = layer_config["units"].get<size_t>();

                    ensure(layer_config["dtype"].get<std::string>() == "float32", "Only float32 weights are supported"); //todo
                    ensure(layer_config["use_bias"].get<bool>(), "Dense without bias is not supported");
                    model->AddLayer(std::make_unique<DenseLayer<T>>(input_size, output_size));
                    mapping[addedLayerIndex] = layer_config["name"];
                    addedLayerIndex++;
                    if (!layer_config["activation"].is_null()) {
                        auto activation_name = layer_config["activation"].get<std::string>();
                        if (activation_name == "relu") {
                            model->AddLayer(std::make_unique<ReLUActivationLayer<T>>());
                        } else if (activation_name == "sigmoid") {
                            model->AddLayer(std::make_unique<SigmoidActivationLayer<T>>());
                        } else if (activation_name == "softmax") {
                            model->AddLayer(std::make_unique<SoftmaxActivationLayer<T>>());
                        } else {
                            throw std::runtime_error("Unknown activation: " + activation_name);
                        }
                        addedLayerIndex++;
                    }
                    last_layer_output_dims = {output_size};
                } else {
                    throw std::runtime_error("Unknown layer: " + layer_name);
                }
            }
            return mapping;
        }

        template <class T>
        static nlohmann::json save_layers(const NeuralNetwork<T>& model, const std::unordered_map<size_t, std::string>& layer_mapping) {
            auto json_layers = nlohmann::json::array();

            for (size_t layer_ind = 0; layer_ind < model.GetLayersCount(); layer_ind++) {
                const auto &layer = model.GetLayer(layer_ind);

                nlohmann::json json_layer = {
                        {"config", {
                               {"name", layer_mapping.at(layer_ind)},
                               {"trainable", true},
                               {"dtype", "float32"} // todo
                       }}
                };

                if (auto dense = dynamic_cast<DenseLayer<T>*>(layer)) {
                    json_layer["class_name"] = "Dense";
                    auto input_size = dense->GetWeightsAndBias().D[0] - 1;
                    auto output_size = dense->GetWeightsAndBias().D[1];

                    if (layer_ind == 0) {
                        json_layer["config"]["batch_input_shape"] = {nullptr, input_size};
                    }
                    json_layer["config"]["units"] = output_size;
                    json_layer["config"]["use_bias"] = true;
                    json_layer["kernel_initializer"] = {
                            {"class_name", "GlorotUniform"},
                            {"config",     {"seed", nullptr}}
                    };
                    json_layer["bias_initializer"] = {
                            {"class_name", "Zeros"},
                            {"config",     {}}
                    };
                    json_layer["kernel_regularizer"] = nullptr;
                    json_layer["bias_regularizer"] = nullptr;
                    json_layer["activity_regularizer"] = nullptr;
                    json_layer["kernel_constraint"] = nullptr;
                    json_layer["bias_constraint"] = nullptr;
                } else if (auto sigmoid = dynamic_cast<SigmoidActivationLayer<T>*>(layer)) {
                    json_layer["class_name"] = "Activation";
                    json_layer["config"]["activation"] = "sigmoid";
                } else if (auto softmax = dynamic_cast<SoftmaxActivationLayer<T>*>(layer)) {
                    json_layer["class_name"] = "Activation";
                    json_layer["config"]["activation"] = "softmax";
                } else {
                    throw std::runtime_error("Can't save layer: " + layer->GetName());
                }

                json_layers.push_back(json_layer);
            }

            nlohmann::json result = {
                    {"class_name", "Sequential"},
                    {"config", {
                              {"name", "sequential_1"},
                              {"layers", json_layers}
                         }
                    },
            };
            return result;
        }

        template<class T>
        static void load_weights(const HighFive::Group &weights,
                                 const std::unordered_map<size_t, std::string> &layers_mapping,
                                 NeuralNetwork<T> *model) {
            for (const auto&[id, name]: layers_mapping) {
                auto layer = model->GetLayer(id);
                if (auto dense = dynamic_cast<DenseLayer<T>*>(layer)) {
                    const auto &dense_group = weights.getGroup(name).getGroup(name);
                    const auto &biasDataset = dense_group.getDataSet("bias:0");
                    const auto &kernelDataset = dense_group.getDataSet("kernel:0");

                    const auto &biasDimensions = biasDataset.getDimensions();
                    const auto &kernelDimensions = kernelDataset.getDimensions();

                    ensure(biasDimensions.size() == 1);
                    ensure(kernelDimensions.size() == 2);
                    ensure(biasDimensions[0] == static_cast<size_t>(dense->GetWeightsAndBias().D[1]));
                    ensure(kernelDimensions[0] == static_cast<size_t>(dense->GetWeightsAndBias().D[0] - 1));
                    ensure(kernelDimensions[1] == static_cast<size_t>(dense->GetWeightsAndBias().D[1]));

                    std::vector<std::vector<float>> kernelWeights; // todo use T
                    kernelDataset.read(kernelWeights);
                    std::vector<float> biasWeights;
                    biasDataset.read(biasWeights);

                    auto weights_and_bias = Tensor<double>::filled({
                                                                           static_cast<int>(kernelDimensions[0] + 1),
                                                                           static_cast<int>(kernelDimensions[1])
                                                                   }, arma::fill::zeros);

                    for (size_t i = 0; i < kernelDimensions[0]; i++) {
                        for (size_t j = 0; j < kernelDimensions[1]; j++) {
                            weights_and_bias.Values().row(i).col(j) = kernelWeights[i][j];
                        }
                    }
                    for (size_t i = 0; i < biasDimensions[0]; i++) {
                        weights_and_bias.Values().row(kernelDimensions[0]).col(i) = biasWeights[i];
                    }

                    dense->SetWeightsAndBias(weights_and_bias);
                } else {
                    throw std::runtime_error("Can't assign weights for layer: " + layer->GetName());
                }
            }
        }

        template<class T>
        static std::unordered_map<size_t, std::string> save_weights(const NeuralNetwork<T> &model,
                                                                    HighFive::Group weights) {
            std::unordered_map<size_t, std::string> mapping;
            std::vector<std::string> layer_with_weight_names;
            for (size_t layer_ind = 0; layer_ind < model.GetLayersCount(); layer_ind++) {
                const auto& layer = model.GetLayer(layer_ind);
                if (auto dense = dynamic_cast<DenseLayer<T>*>(layer)) {

                    std::vector<std::vector<float>> kernelWeights; // todo use T
                    std::vector<float> biasWeights;

                    const auto& weights_and_bias = dense->GetWeightsAndBias();

                    kernelWeights.resize(weights_and_bias.D[0] - 1);
                    for (size_t i = 0; i < static_cast<size_t>(weights_and_bias.D[0] - 1); i++) {
                        kernelWeights[i].resize(weights_and_bias.D[1]);
                        for (size_t j = 0; j < static_cast<size_t>(weights_and_bias.D[1]); j++) {
                            kernelWeights[i][j] = weights_and_bias.Values()(i, j);
                        }
                    }
                    biasWeights.resize(weights_and_bias.D[1]);
                    for (size_t i = 0; i < static_cast<size_t>(weights_and_bias.D[1]); i++) {
                        biasWeights[i] = weights_and_bias.Values()(weights_and_bias.D[0] - 1, i);
                    }

                    std::string name = "dense_" + std::to_string(layer_ind);
                    mapping[layer_ind] = name;
                    layer_with_weight_names.push_back(name);
                    auto dense_group = weights.createGroup(name);
                    WriteAttribute(dense_group, "weight_names", {name + "/kernel:0", name + "/bias:0"});
                    auto dense_subgroup = dense_group.createGroup(name);
                    dense_subgroup.createDataSet("bias:0", biasWeights);
                    dense_subgroup.createDataSet("kernel:0", kernelWeights);
                } else if (auto sigmoid = dynamic_cast<SigmoidActivationLayer<T>*>(layer)) {
                    mapping[layer_ind] = "sigmoid_" + std::to_string(layer_ind);
                } else if (auto softmax = dynamic_cast<SoftmaxActivationLayer<T>*>(layer)) {
                    mapping[layer_ind] = "softmax_" + std::to_string(layer_ind);
                } else {
                    throw std::runtime_error("Can't save weights for layer: " + layer->GetName());
                }
            }

            WriteAttribute(weights, "layer_names", layer_with_weight_names);

            return mapping;
        }
    };
}