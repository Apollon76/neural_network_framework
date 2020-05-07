#include <iomanip>
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <src/neural_network.hpp>
#include <src/tensor.hpp>
#include <benchmarks/cases/mnist/config.hpp>
#include <benchmarks/cases/cifar/config.hpp>

class Action {
public:
    explicit Action(const std::string &action_description) :
            begin(std::chrono::steady_clock::now()) {
        std::cerr << action_description << "..." << std::flush;
    }

    Action(const Action &) = delete;

    Action(const Action &&) = delete;

    Action &operator=(const Action &) = delete;

    Action &operator=(const Action &&) = delete;

    [[nodiscard]] long long GetSpentMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
    }

    ~Action() {
        auto end = std::chrono::steady_clock::now();
        std::cerr << "Done ("
                  << FormatDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin))
                  << ")" << std::endl;
    }

private:
    static std::string FormatDuration(const std::chrono::duration<int64_t, std::milli> &duration) {
        int total = static_cast<int>(duration.count());
        int ms = total % 1000;
        total /= 1000;
        int s = total % 60;
        total /= 60;
        int m = total % 60;
        total /= 60;
        int h = total;

        char buffer[100];
        sprintf(buffer, "%02d:%02d:%02d.%03d", h, m, s, ms);
        return std::string(buffer);
    }

    decltype(std::chrono::steady_clock::now()) begin;
};


int main(int argc, char **argv) {
    cxxopts::Options options("Run model fitting and evaluation");
    options.add_options()("test-name", "name of benchmark to run (./cases/*)", cxxopts::value<std::string>());
    options.add_options()("data-path", "path to data", cxxopts::value<std::string>());
    options.add_options()("epochs", "number of epochs to fit", cxxopts::value<int>()->default_value("5"));
    options.add_options()("batch-size", "batch size used while fitting", cxxopts::value<int>()->default_value("32"));
    auto parsed_args = options.parse(argc, argv);
    ensure(parsed_args["test-name"].count(), "test-name is required arg");
    ensure(parsed_args["data-path"].count(), "data-path is required arg");

    auto test_name = parsed_args["test-name"].as<std::string>();
    auto data_path = parsed_args["data-path"].as<std::string>();
    auto epochs = parsed_args["epochs"].as<int>();
    auto batch_size = parsed_args["batch-size"].as<int>();

    std::shared_ptr<Config> config;
    if (test_name == "mnist") {
        config = std::dynamic_pointer_cast<Config>(std::make_shared<benchmarks::MnistConfig>());
    } else if (test_name == "cifar") {
        config = std::dynamic_pointer_cast<Config>(std::make_shared<benchmarks::CifarConfig>());
    } else {
        throw std::runtime_error("Unknown test: " + test_name);
    }

    Config::DataType data;
    {
        Action action("Loading data from " + data_path);
        data = config->LoadData(data_path);
    }
    Tensor<float> x_train, y_train, x_test, y_test;
    std::tie(x_train, y_train, x_test, y_test) = data;

    NeuralNetwork<float> model;
    {
        Action action("Building model");
        model = config->BuildModel();
    }

    struct Metric {
        double train_score;
        double test_score;
        double train_loss;
        double test_loss;
    };
    std::unordered_map<int, Metric> metrics;
    long long totalMsSpent = 0;
    {
        auto callback = std::make_shared<EpochCallback<float>>(
                [&x_test, &y_test, &y_train, &config, &metrics, epochs](const INeuralNetwork<float> *model, int epoch) {
                    return [&x_test, &y_test, &y_train, &config, &metrics, model, epoch, epochs](const Tensor<float>& train_prediction, double train_loss) {
                        auto test_prediction = model->Predict(x_test);
                        Metric metric{};
                        metric.train_score = config->GetScore(y_train, train_prediction);
                        metric.test_score = config->GetScore(y_test, test_prediction);
                        metric.train_loss = train_loss;
                        metric.test_loss = model->GetLoss()->GetLoss(test_prediction, y_test);
                        metrics[epoch] = metric;
                        std::cerr << "Epoch: " << epoch + 1 << "/" << epochs
                                  << " train score: " << metric.train_score << " train loss: " << metric.train_loss
                                  << " test score: " << metric.test_score << " test loss: " << metric.test_loss
                                  << std::endl;
                        return CallbackSignal::Continue;
                    };
                });
        Action action("Fitting model for " + std::to_string(epochs) + " epochs");
        model.Fit(x_train, y_train, epochs, batch_size, true, callback);
        totalMsSpent = action.GetSpentMs();
    }

    double score;
    {
        Action action("Evaluating model on test data");
        score = config->GetScore(y_test, model.Predict(x_test));
    }
    std::cerr << "Evaluation result: " << score << std::endl;

    nlohmann::json result = {
            {"epochs", epochs},
            {"batch_size", batch_size},
            {"time_ms_total", totalMsSpent},
            {"metrics", {}}
    };

    for (const auto& [epoch, metric] : metrics) {
        result["metrics"]["loss"]["train"].push_back({epoch, metric.train_loss});
        result["metrics"]["loss"]["test"].push_back({epoch, metric.test_loss});
        result["metrics"][config->GetScoreName()]["train"].push_back({epoch, metric.train_score});
        result["metrics"][config->GetScoreName()]["test"].push_back({epoch, metric.train_score});
    }

    {
        std::ofstream result_file("/tmp/results.json");
        result_file << result.dump() << std::endl;
    }

    return 0;
}