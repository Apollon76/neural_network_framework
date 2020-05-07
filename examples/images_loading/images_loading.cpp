#include <memory>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include <src/io/filesystem.hpp>
#include <src/io/img.hpp>
#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/layers/activations.hpp>
#include <src/optimizer.hpp>
#include <src/utils.hpp>
#include <src/layers/dense.hpp>
#include <src/tensor.hpp>
#include <src/callbacks/progress_bar_callback.hpp>

std::tuple<Tensor<double>, Tensor<double>> LoadMnistPng(const std::string &path) {
    std::cout << "Loading png mnist dataset from " << path << std::endl;
    Timer timer("Load of " + path, true);
    auto paths = nn_framework::io::Filesystem::ListFiles(path);
    nn_framework::io::ImgReader reader(paths);

    auto x = Tensor<unsigned char>::filled({(int) paths.size(), 28 * 28}, arma::fill::zeros);
    auto y = Tensor<unsigned char>::filled({(int) paths.size(), 1}, arma::fill::zeros);
    int pos = 0;
    for (const auto&[name, img] : reader.LoadDataWithNames<arma::u8>()) {
        ensure(img.n_slices == 1, "mnist images should be grayscale");
        x.Values().row(pos) = img.slice(0).as_row();
        y.Values().row(pos) = std::stoi(name.substr(10, 1)); // 000000-numX.png
        pos++;
    }
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {x.ConvertTo<double>(), y_one_hot.ConvertTo<double>()};
}

NeuralNetwork<double> BuildMnistNN() {
    auto neural_network = NeuralNetwork<double>(std::make_unique<AdamOptimizer<double>>(),
                                                std::make_unique<CategoricalCrossEntropyLoss<double>>());
    neural_network
            .AddLayer(std::make_unique<DenseLayer<double>>(784, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer<double>>())
            .AddLayer(std::make_unique<DenseLayer<double>>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());

    return neural_network;
}

void MnistPng(const std::string &data_path) {
    auto[x_train, y_train] = LoadMnistPng(data_path + "/train");
    auto[x_test, y_test] = LoadMnistPng(data_path + "/test");

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    std::cout << "Start mnist neural network..." << std::endl;
    auto model = BuildMnistNN();
    model.Fit(x_train, y_train, 40, NoBatches, false, std::make_shared<ProgressBarCallback<double>>());

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(model.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(model.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    cxxopts::Options options("loading images example (on mnist images)");
    options.add_options()("data-path", "path to data", cxxopts::value<std::string>());
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["data-path"].as<std::string>();

    MnistPng(data_path + "/mnist-png");

    return 0;
}