#include <iostream>
#include <src/io/csv.hpp>
#include <src/data_processing/data_utils.hpp>
using namespace std;

std::tuple<arma::mat, arma::mat> LoadMnist(const std::string& path) {
    cout << "Loading mnist dataset from " << path << endl;
    Timer timer("Load of "+ path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = CreateMatrix(csv_data_provider.LoadData<int>());
    auto X = data.tail_cols(data.n_cols - 1);
    auto y = data.head_cols(1);
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {arma::conv_to<arma::mat>::from(X), arma::conv_to<arma::mat>::from(y_one_hot)};
}

arma::mat LoadMnistX(const std::string& path) {
    Timer timer("Load of "+ path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = CreateMatrix(csv_data_provider.LoadData<int>());
    return arma::conv_to<arma::mat>::from(data);
}