#include <vector>
#include <memory>
#include <armadillo>
#include <glog/logging.h>

struct Gradients {
    arma::mat input_gradients;
    std::vector<arma::mat> layer_gradients;
};

std::string FormatDimensions(const arma::mat &mat) {
    return std::to_string(mat.n_rows) + "x" + std::to_string(mat.n_cols);
}

class ILayer {
public:
    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual arma::mat Apply(const arma::mat &) const = 0;

    [[nodiscard]] virtual Gradients PullGradientsBackward(
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const = 0;

    virtual void ApplyGradients(const std::vector<arma::mat> &gradients) = 0;

    virtual ~ILayer() = default;
};

class DenseLayer : public ILayer {
public:
    DenseLayer(arma::uword n_rows, arma::uword n_cols) : weights(arma::randu(n_rows, n_cols)),
                                                         bias(arma::randu(1, n_cols)) {
    }

    [[nodiscard]] const arma::mat &GetWeights() const {
        return weights;
    }

    [[nodiscard]] std::string ToString() const override {
        return "Dense [" + FormatDimensions(weights) + "]";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        return input * weights + bias;
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const override {
        LOG(INFO) << "Pull gradients for dense layer: "
                  << "input=[" + FormatDimensions(input) + "], "
                  << "output_gradients=[" + FormatDimensions(output_gradients) + "], "
                  << "weights=[" + FormatDimensions((weights)) + "], "
                  << "bias=[" + FormatDimensions((bias)) + "]";
        return Gradients{
                output_gradients * arma::trans(weights),
                {
                        arma::kron(output_gradients, arma::trans(input)),
                        output_gradients
                }
        };
    }

    void ApplyGradients(const std::vector<arma::mat> &gradients) override {
        LOG(INFO) << "Apply gradients for dense layer: "
                  << "gradients[0]=[" + FormatDimensions((gradients[0])) + "], "
                  << "gradients[1]=[" + FormatDimensions((gradients[1])) + "], "
                  << "weights=[" + FormatDimensions((weights)) + "], "
                  << "bias=[" + FormatDimensions((bias)) + "]";
        weights += gradients[0];
        bias += gradients[1];
    }

private:
    arma::mat weights;
    arma::mat bias;
};

class SigmoidActivationLayer : public ILayer {
public:
    [[nodiscard]] std::string ToString() const override {
        return "SigmoidActivation";
    }

    [[nodiscard]] arma::mat Apply(const arma::mat &input) const override {
        auto expInput = arma::exp(input);
        return expInput / (expInput + 1);
    }

    [[nodiscard]] Gradients PullGradientsBackward(
            const arma::mat &,
            const arma::mat &output_gradients
    ) const override {
        auto expOutput = arma::exp(output_gradients);
        return Gradients{
                expOutput / arma::square(expOutput + 1),
                std::vector<arma::mat>()
        };
    }

    void ApplyGradients(const std::vector<arma::mat> &) override {

    }

private:
};

class IOptimizer {
public:
    [[nodiscard]] virtual std::vector<arma::mat> GetGradientStep(const std::vector<arma::mat> &gradients) const = 0;
};

class Optimizer : public IOptimizer {
public:
    explicit Optimizer(double _learning_rate) : learning_rate(_learning_rate) {}

    [[nodiscard]] std::vector<arma::mat> GetGradientStep(const std::vector<arma::mat> &gradients) const override {
        auto result = std::vector<arma::mat>(gradients.size());
        for (size_t i = 0; i < gradients.size(); i++) {
            result[i] = -gradients[i] * learning_rate;
        }
        return result;
    }

private:
    double learning_rate;
};

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
        LOG(INFO) << "Fitting neural network...";
        std::vector<arma::mat> inter_outputs = {input};
        for (auto &&layer : layers) {
            LOG(INFO) << "Fit forward layer: " << layer->ToString();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        std::vector<std::vector<arma::mat>> layer_gradients = {};
        std::vector<arma::mat> output_gradients = {2 * (inter_outputs.back() - output)};
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
            LOG(INFO) << "Propagate gradients backward for layer: " << layers[i]->ToString();
            auto gradients = layers[i]->PullGradientsBackward(inter_outputs[i], output_gradients.back());
            auto gradients_to_apply = optimizer->GetGradientStep(gradients.layer_gradients);
            layers[i]->ApplyGradients(gradients_to_apply);
            LOG(INFO) << "Old output gradients: [" << FormatDimensions(output_gradients.back()) << "], "
                      << "new output gradients: [" << FormatDimensions(gradients.input_gradients) << "]";
            output_gradients.push_back(gradients.input_gradients);
        }
        LOG(INFO) << "Fitting finished";
    }

    [[nodiscard]] arma::mat Predict(const arma::mat &input) const {
        LOG(INFO) << "Predict with neural network...";
        std::vector<arma::mat> inter_outputs = {input};
        for (auto &&layer : layers) {
            LOG(INFO) << "Fit forward layer: " << layer->ToString();
            inter_outputs.push_back(layer->Apply(inter_outputs.back()));
        }
        return inter_outputs.back();
    }


private:
    std::vector<std::unique_ptr<ILayer>> layers;
    std::unique_ptr<IOptimizer> optimizer;
};

int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "Start example neural network...";
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.01));
    neural_network.AddLayer(std::make_unique<DenseLayer>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(3, 1));
    auto input = arma::mat({{1, 0}});
    auto output = arma::mat({0});
    output.set_size(1, 1);
    for (int i = 0; i<  100; i++) {
        neural_network.Predict(input).print();
        neural_network.Fit(input, output);
    }
    return 0;
}