#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include <src/io/csv.hpp>
#include <src/io/filesystem.hpp>
#include <src/io/img.hpp>
#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/layers/activations.hpp>
#include <src/optimizer.hpp>
#include <src/utils.hpp>
#include <src/layers/dense.hpp>
#include <src/layers/dropout.hpp>
#include <src/layers/flatten.hpp>
#include <src/layers/convolution2d.hpp>
#include <src/callbacks/performance_metrics_callback.hpp>
#include <src/callbacks/logging_callback.hpp>
#include <src/os_utils.hpp>
#include <src/tensor.hpp>

void GenerateInputs(Tensor<double> &inputs, Tensor<double> &outputs) {
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_int_distribution<> coin(0, 1);
    std::uniform_real_distribution real;
    auto inputs_vector = std::vector<std::vector<double>>();
    auto outputs_vector = std::vector<std::vector<double>>();
    for (int i = 0; i < 100; i++) {
        auto type = coin(gen);
        auto radius = 5 * type + 1;
        auto angle = real(gen) * 2 * M_PI;
        auto x = radius * cos(angle) + real(gen) * 0.1;
        auto y = radius * sin(angle) + real(gen) * 0.1;
        inputs_vector.push_back({x, y});
        outputs_vector.push_back({(double) type});
    }
    inputs = Tensor<double>::fromVector(inputs_vector);
    outputs = Tensor<double>::fromVector(outputs_vector);
}

void Sample() {
    DLOG(INFO) << "Start example neural network...";
    auto neural_network = NeuralNetwork<double>(std::make_unique<Optimizer<double>>(0.01),
                                                std::make_unique<MSELoss<double>>());
    neural_network.AddLayer(std::make_unique<DenseLayer<double>>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer<double>>());
    neural_network.AddLayer(std::make_unique<DenseLayer<double>>(3, 1));
    Tensor<double> inputs, outputs;
    GenerateInputs(inputs, outputs);
    for (int i = 0; i < 10000; i++) {
        std::cout << "Loss: " << neural_network.FitOneIteration(inputs, outputs) << std::endl;
    }
    std::cout << arma::join_rows(neural_network.Predict(inputs).Values(), outputs.Values()) << std::endl;
    std::cout << neural_network.ToString() << std::endl;
}

NeuralNetwork<double> BuildMnistNN(std::unique_ptr<IOptimizer<double>> optimizer) {
    auto neural_network = NeuralNetwork<double>(std::move(optimizer),
                                                std::make_unique<CategoricalCrossEntropyLoss<double>>());
    neural_network
            .AddLayer(std::make_unique<DenseLayer<double>>(784, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer<double>>())
            .AddLayer(std::make_unique<DenseLayer<double>>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());

    return neural_network;
}

NeuralNetwork<double> BuildMnistNNConv(std::unique_ptr<IOptimizer<double>> optimizer) {
    auto conv_filters = 2;
    auto img_size = 28;
    auto neural_network = NeuralNetwork<double>(std::move(optimizer),
                                                std::make_unique<CategoricalCrossEntropyLoss<double>>());
    neural_network
            .AddLayer(std::make_unique<Convolution2dLayer<double>>(1, conv_filters, 3, 3, ConvolutionPadding::Same))
            .AddLayer(std::make_unique<ReLUActivationLayer<double>>())
            .AddLayer(std::make_unique<DropoutLayer<double>>(0.1))
            .AddLayer(std::make_unique<FlattenLayer<double>>(std::vector<int>{0, conv_filters, img_size, img_size}))
            .AddLayer(std::make_unique<DenseLayer<double>>(img_size * img_size * conv_filters, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer<double>>())
            .AddLayer(std::make_unique<DenseLayer<double>>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());

    return neural_network;
}

template<typename T>
void FitNN(NeuralNetwork<T> *neural_network,
           int epochs,
           const Tensor<T> &x_train,
           const Tensor<T> &y_train,
           const std::optional<Tensor<T>> &x_test = std::nullopt,
           const std::optional<Tensor<T>> &y_test = std::nullopt) {
    Timer timer("Fitting ");
    for (int i = 0; i < epochs; i++) {
        auto loss = neural_network->FitOneIteration(x_train, y_train);
        if (i % 5 == 0) {
            auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(x_train), y_train);
            if (x_test.has_value()) {
                auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(*x_test),
                                                                                *y_test);
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score
                          << " Test score: " << test_score << std::endl;
            } else {
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score
                          << std::endl;
            }
        } else {
            std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << std::endl;
        }
    }
}

void DigitRecognizer(const std::string &data_path, const std::string &output,
                     std::unique_ptr<IOptimizer<double>> optimizer) {
    auto[x_train, y_train] = LoadMnist(data_path + "/kaggle-digit-recognizer/train.csv", true);
    auto x_test = LoadMnistX(data_path + "/kaggle-digit-recognizer/test.csv", true);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;
    LOG(INFO) << "Start digit-recognizer neural network...";

    auto neural_network = BuildMnistNN(std::move(optimizer));
    FitNN(&neural_network, 40, x_train, y_train);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    std::cout << "Final train score: " << train_score << std::endl;

    arma::ucolvec predictions = arma::index_max(neural_network.Predict(x_test).Values(), 1);
    std::string predictions_path = data_path + "/kaggle-digit-recognizer/" + output;
    nn_framework::io::CsvWriter writer(predictions_path);
    writer.WriteRow({"ImageId", "Label"});
    for (arma::u64 i = 0; i < predictions.n_rows; i++) {
        writer.WriteRow({i + 1, predictions.at(i, 0)});
    }
    std::cout << "Predictions written to " << predictions_path << std::endl;
}

void DigitRecognizerConv(const std::string &data_path, const std::string &output,
                         std::unique_ptr<IOptimizer<double>> optimizer) {
    auto[x_train_raw, y_train] = LoadMnist(data_path + "/kaggle-digit-recognizer/train.csv", true);
    auto x_test_raw = LoadMnistX(data_path + "/kaggle-digit-recognizer/test.csv", true);

    auto train_reshaper = FlattenLayer<double>({x_train_raw.D[0], 1, 28, 28});
    auto test_reshaper = FlattenLayer<double>({x_test_raw.D[0], 1, 28, 28});

    auto x_train = train_reshaper.PullGradientsBackward(Tensor<double>(), x_train_raw).input_gradients;
    auto x_test = train_reshaper.PullGradientsBackward(Tensor<double>(), x_test_raw).input_gradients;

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;
    LOG(INFO) << "Start digit-recognizer neural network...";

    auto neural_network = BuildMnistNNConv(std::move(optimizer));
    auto callback = EveryNthEpoch<double>(
            10,
            ScoreCallback<double>("train score", [](const Tensor<double> &a,
                                                    const Tensor<double> &b) {
                return nn_framework::scoring::one_hot_accuracy_score(a, b);
            }, x_train, y_train));
//    neural_network.AddCallback<PerformanceMetricsCallback>();
//    neural_network.AddCallback<LoggingCallback>();
    neural_network.Fit(x_train, y_train, 40, 128, true, callback);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    std::cout << "Final train score: " << train_score << std::endl;

    arma::ucolvec predictions = arma::index_max(neural_network.Predict(x_test).Values(), 1);
    std::string predictions_path = data_path + "/kaggle-digit-recognizer/" + output;
    nn_framework::io::CsvWriter writer(predictions_path);
    writer.WriteRow({"ImageId", "Label"});
    for (arma::u64 i = 0; i < predictions.n_rows; i++) {
        writer.WriteRow({i + 1, predictions.at(i, 0)});
    }
    std::cout << "Predictions written to " << predictions_path << std::endl;
}

void DigitRecognizerValidation(const std::string &data_path, std::unique_ptr<IOptimizer<double>> optimizer) {
    auto[x, y] = LoadMnist(data_path + "/kaggle-digit-recognizer/train.csv", true);

    nn_framework::data_processing::TrainTestSplitter<double> splitter(42);
    auto[x_train, y_train, x_test, y_test] = splitter.Split(x, y, 0.7);

    std::cout << "X_train: " << FormatDimensions(x_train) << " y_train: " << FormatDimensions(y_train) << std::endl;
    std::cout << "X_test: " << FormatDimensions(x_test) << " y_test: " << FormatDimensions(y_test) << std::endl;
    LOG(INFO) << "Start digit-recognizer neural network...";

    auto neural_network = BuildMnistNN(std::move(optimizer));
    FitNN<double>(&neural_network, 40, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << std::endl;
    std::cout << "Final validation score: " << test_score << std::endl;
}

void Mnist(const std::string &data_path) {
    auto[x_train, y_train] = LoadMnist(data_path + "/mnist/mnist_train.csv", false);
    auto[x_test, y_test] = LoadMnist(data_path + "/mnist/mnist_test.csv", false);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = BuildMnistNN(std::make_unique<Optimizer<double>>(0.00001));
    FitNN<double>(&neural_network, 20, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

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

void MnistPng(const std::string &data_path, bool scale) {
    auto[x_train, y_train] = LoadMnistPng(data_path + "/mnist-png/train");
    auto[x_test, y_test] = LoadMnistPng(data_path + "/mnist-png/test");

    if (scale) {
        nn_framework::data_processing::Scaler<double> scaler;
        scaler.Fit(x_train);
        x_train = scaler.Transform(x_train);
        x_test = scaler.Transform(x_test);
    }

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = BuildMnistNN(std::make_unique<Optimizer<double>>(0.00001));
    FitNN<double>(&neural_network, 20, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    arma::arma_config cfg;
    std::cout << "blas enabled: " << cfg.blas << std::endl;
    std::cout << "openmp enabled: " << cfg.openmp << std::endl;
    std::cout << "lapack enabled: " << cfg.lapack << std::endl;
    std::cout << "superlu enabled: " << cfg.superlu << std::endl;
    std::cout << "mp_threads enabled: " << cfg.mp_threads << std::endl;

    cxxopts::Options options("nn framework main");
    options.add_options()
            ("d,data", "path to data", cxxopts::value<std::string>()->default_value("../.."));
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["data"].as<std::string>();

    //MnistPng(data_path + "/data", false);
    //Mnist(data_path + "/data");
//    DigitRecognizerValidation(data_path + "/data", std::make_unique<RMSPropOptimizer<double>>(0.01));
    DigitRecognizerConv(data_path + "/data", "output", std::make_unique<RMSPropOptimizer<double>>());
//    DigitRecognizer(data_path + "/data", data_path + "/output", std::make_unique<RMSPropOptimizer<double>>(0.01));
    return 0;
}