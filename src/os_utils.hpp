#pragma once

#include <iostream>
#include <src/io/csv.hpp>
#include <src/data_processing/data_utils.hpp>
#include <src/tensor.hpp>

using namespace std;

std::tuple<Tensor<double>, Tensor<double>> LoadMnist(const std::string &path) {
    cout << "Loading mnist dataset from " << path << endl;
    Timer timer("Load of " + path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
    auto X = Tensor<int>({data.D[0], data.D[1] - 1}, data.Values().tail_cols(data.D[1] - 1));
    auto y = Tensor<int>({data.D[0], 1}, data.Values().head_cols(1));
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {X.ConvertTo<double>(), y_one_hot.ConvertTo<double>()};
}

Tensor<double> LoadMnistX(const std::string &path) {
    Timer timer("Load of " + path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
    return data.ConvertTo<double>();
}