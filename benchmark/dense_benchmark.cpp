#include <iostream>
#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include "src/io/csv.hpp"
#include <src/data_processing/data_utils.hpp>
#include <src/scoring/scoring.hpp>
#include "neural_network.hpp"
#include "src/layers/activations.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include "layers/dense.hpp"
using namespace std;

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    cxxopts::Options options("nn framework main");

    options.add_options()
            ("d,test_data", "path to test data")
            ;
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["data"].as<std::string>();

    cout << "Hello world!\n";
    cout << "Here is data path: " << data_path << '\n';
    return 0;
}