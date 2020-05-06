#include <cxxopts.hpp>
#include <src/io/csv.hpp>
#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/optimizer.hpp>
#include <src/utils.hpp>
#include <src/os_utils.hpp>
#include <src/tensor.hpp>
#include <src/serialization/hdf5_serialization.hpp>

void load_model_and_evaluate(const std::string& model_path, const std::string& data_path) {
    auto neural_network = nn_framework::serialization::hdf5::Hdf5Serializer::LoadModel(model_path);

    auto[x_train, y_train] = LoadMnist(data_path + "/mnist/mnist_train.csv", false);
    auto[x_test, y_test] = LoadMnist(data_path + "/mnist/mnist_test.csv", false);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Train score: " << train_score << " test score: " << test_score << std::endl;

    ensure(test_score > 0.8);
}

template<class T>
void FitNN(NeuralNetwork<T> *neural_network, int epochs, const Tensor<T> &x_train, const Tensor<T> &y_train,
           const Tensor<T> &x_test, const Tensor<T> &y_test) {
    Timer timer("Fitting ", true);
    for (int i = 0; i < epochs; i++) {
        auto loss = neural_network->Fit(x_train, y_train);
        if (i % 5 == 0) {
            auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(x_train), y_train);
            auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(x_test), y_test);
            std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score
                      << " Test score: " << test_score << std::endl;
        } else {
            std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << std::endl;
        }
    }
}

void train_model_and_save(const std::string& save_to_path, const std::string& data_path) {
    auto[x_train, y_train] = LoadMnist(data_path + "/mnist/mnist_train.csv", false);
    auto[x_test, y_test] = LoadMnist(data_path + "/mnist/mnist_test.csv", false);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    auto neural_network = NeuralNetwork<double>(std::make_unique<AdamOptimizer<double>>(),
                                                std::make_unique<CategoricalCrossEntropyLoss<double>>());
    neural_network
            .AddLayer<DenseLayer>(784, 100)
            .AddLayer<SigmoidActivationLayer>()
            .AddLayer<DenseLayer>(100, 10)
            .AddLayer<SoftmaxActivationLayer>();

    FitNN<double>(&neural_network, 10, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;

    nn_framework::serialization::hdf5::Hdf5Serializer::SaveModel(neural_network, save_to_path);
}

int main(int argc, char **argv) {
    cxxopts::Options options("Example on saving and loading model in keras-compatible hdf5 format");
    options.add_options()("data-path", "path to data", cxxopts::value<std::string>());
    options.add_options()("model-file", "path to model", cxxopts::value<std::string>());
    options.add_options()("mode", "load|save", cxxopts::value<std::string>());
    auto parsed_args = options.parse(argc, argv);
    ensure(parsed_args["data-path"].count(), "data-path is required arg");
    ensure(parsed_args["model-file"].count(), "model-file is required arg");
    ensure(parsed_args["mode"].count(), "mode is required arg");
    auto data_path = parsed_args["data-path"].as<std::string>();
    auto model_file = parsed_args["model-file"].as<std::string>();
    auto mode = parsed_args["mode"].as<std::string>();

    if (mode == "load") {
        load_model_and_evaluate(model_file, data_path);
    } else if (mode == "save") {
        train_model_and_save(model_file, data_path);
    } else {
        throw std::runtime_error("Unknown mode: " + mode);
    }

    return 0;
}