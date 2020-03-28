#pragma once

#include <armadillo>
#include <utility>
#include <vector>
#include "utils.hpp"

using TensorDimensions = std::vector<int>;

template<typename T>
class Tensor {
public:
    Tensor()
            : D(),
              values() {
    }

    Tensor(TensorDimensions _dimensions, arma::Mat<T> _values)
            : D(std::move(_dimensions)),
              values(std::move(_values)) {
    }

    template<typename TNew>
    Tensor<TNew> ConvertTo() const {
        return Tensor<TNew>(D, arma::conv_to<arma::Mat<TNew>>::from(values));
    }

    const T &at(int x, int y) const {
        return values.at(x, y);
    }

    T &at(int x, int y) {
        return values.at(x, y);
    }

    [[nodiscard]] arma::Mat<T> &Values() {
        return values;
    }

    [[nodiscard]] const arma::Mat<T> &Values() const {
        return values;
    }

    [[nodiscard]] int TotalElements() const {
        return std::accumulate(D.begin(), D.end(), 1, [](int a, int b) { return a * b; });
    }

    [[nodiscard]] int Rank() const {
        return D.size();
    }

    [[nodiscard]] int BatchCount() const {
        return D[0];
    }

    TensorDimensions D;
private:
    arma::Mat<T> values;
};


template<typename T>
class TensorInitializer {
public:
    TensorInitializer(T x) // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
            : dimensions(),
              values(arma::Mat<T>({x})) {
    }

    TensorInitializer(std::initializer_list<TensorInitializer<T>> list) : dimensions(), values() {
        auto other = std::optional<TensorDimensions>();
        auto valueList = std::vector<arma::Row<T>>();
        for (auto &&item : list) {
            ensure(!other.has_value() || item.dimensions == other.value(),
                   "Nested tensors must have equal dimensions");
            other = item.dimensions;
            valueList.push_back(item.values.as_row());
        }
        ensure(other.has_value(), "Tensor dimensions must be non-empty");
        values = arma::Mat<T>(list.size(), valueList[0].size());
        for (size_t i = 0; i < list.size(); i++) {
            values.row(i) = valueList[i];
        }
        dimensions = other.value();
        dimensions.insert(dimensions.begin(), list.size());
    }

    Tensor<T> R() const {
        return Tensor<T>(dimensions, values);
    }

    TensorDimensions dimensions;
    arma::Mat<T> values;
};

template<typename T, typename FillType>
Tensor<T> TensorFill(TensorDimensions d, FillType fill) {
    static_assert(std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_ones>>::value != 0 ||
                  std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_zeros>>::value != 0 ||
                  std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randn>>::value != 0 ||
                  std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randu>>::value != 0);
    auto batch_size = d[0];
    auto other_size = std::accumulate(d.begin() + 1, d.end(), 1,
                                      [](int a, int b) { return a * b; });
    return Tensor<T>(d, arma::Mat<T>(batch_size, other_size, fill));
}

template<typename T>
Tensor<T> CreateMatrix(const std::vector<std::vector<T>> &values) {
    auto mat = TensorFill<T>({(int) values.size(), (int) values[0].size()}, arma::fill::zeros);
    for (int i = 0; i < (int) values.size(); i++) {
        for (int s = 0; s < (int) values[0].size(); s++) {
            mat.Values().at(i, s) = values[i][s];
        }
    }
    return mat;
}

template<typename T>
std::string FormatDimensions(const Tensor<T> &t) {
    auto d = t.D;
    auto result = std::string();
    for (size_t i = 0; i < d.size(); i++) {
        if (i != 0) {
            result += " x ";
        }
        result += std::to_string(d[i]);
    }
    return result;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
    return os << "Tensor(" << typeid(T).name() << ")"
              << "[" << FormatDimensions(t) << "]" << std::endl
              << t.Values() << std::endl;
}
