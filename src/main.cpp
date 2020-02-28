#include <vector>
#include <memory>
#include <armadillo>

class ILayer {
public:
    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual arma::mat Predict(arma::mat input) const = 0;
};

class DenseLayer : public ILayer {
public:
    DenseLayer(arma::uword n_rows, arma::uword n_cols) : weights(arma::randu(n_rows, n_cols)) {
    }

    [[nodiscard]] const arma::mat &GetWeights() const {
        return weights;
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream output;
        arma::arma_ostream::print(output, weights, true);
        return "Dense:\n" + output.str();
    }

    [[nodiscard]] arma::mat Predict(arma::mat input) const override {
        return input * weights;
    }

private:
    arma::mat weights;
};

class SigmoidActivationLayer : public ILayer {
public:
    [[nodiscard]] std::string ToString() const override {
        return "SigmoidActivation";
    }

    ~SigmoidActivationLayer() {
        std::cout << "Destructing!!" << std::endl;
    }

    [[nodiscard]] arma::mat Predict(arma::mat input) const override {
        auto expInput = arma::exp(input);
        return expInput / (expInput + arma::ones(arma::size(input)));
    }

private:
};

class NeuralNetwork : public ILayer {
public:
    NeuralNetwork() : layers() {}

    void AddLayer(std::unique_ptr<ILayer> layer) {
        layers.emplace_back(std::move(layer));
    }

    [[nodiscard]] std::string ToString() const override {
        std::stringstream output;
        output << "Layers:\n";
        for (auto &layer : layers) {
            output << layer->ToString() << std::endl;
        }
        return output.str();
    }

    [[nodiscard]] arma::mat Predict(arma::mat input) const override {
        auto output = input;
        for (auto &&layer : layers) {
            output = layer->Predict(output);
        }
        return output;
    }


private:
    std::vector<std::unique_ptr<ILayer>> layers;
};

int main(int argc, char **argv) {
    auto neural_network = NeuralNetwork();
    neural_network.AddLayer(std::make_unique<DenseLayer>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(3, 1));
    std::cout << neural_network.ToString() << std::endl;
    auto input = arma::mat({{1, 2}});
    auto output = neural_network.Predict(input);
    input.print("Input:");
    output.print("Output:");
    
    return 0;
}