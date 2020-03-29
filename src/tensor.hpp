#pragma once

#include <armadillo>
#include <utility>
#include <vector>
#include <src/utils.hpp>

using TensorDimensions = std::vector<int>;

template<typename T>
class TensorInitializer {
public:
    TensorInitializer(T x) // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
            : D(),
              values(arma::Mat<T>({x})) {
    }

    TensorInitializer(std::initializer_list<TensorInitializer<T>> list) : D(), values() {
        auto other = std::optional<TensorDimensions>();
        auto valueList = std::vector<arma::Row<T>>();
        for (auto &&item : list) {
            ensure(!other.has_value() || item.D == other.value(),
                   "Nested tensors must have equal dimensions");
            other = item.D;
            valueList.push_back(item.values.as_row());
        }
        ensure(other.has_value(), "Tensor dimensions must be non-empty");
        values = arma::Mat<T>(list.size(), valueList[0].size());
        for (size_t i = 0; i < list.size(); i++) {
            values.row(i) = valueList[i];
        }
        D = other.value();
        D.insert(D.begin(), list.size());
    }

    TensorDimensions D;
    arma::Mat<T> values;
};

template<typename T>
arma::field<arma::Mat<T>> createValuesContainer(TensorDimensions d) {
    ensure(d.size() <= 5);
    if (d.size() <= 2) {
        return arma::field<arma::Mat<T>>(1);
    } else if (d.size() == 3) {
        return arma::field<arma::Mat<T>>(d[0]);
    } else if (d.size() == 4) {
        return arma::field<arma::Mat<T>>(d[0], d[1]);
    } else if (d.size() == 5) {
        return arma::field<arma::Mat<T>>(d[0], d[1], d[2]);
    } else {
        throw std::logic_error("too many dimensions");
    }
}

template<typename T>
class Tensor {
public:
    Tensor()
            : D(),
              values(arma::field<arma::Mat<T>>(1)) {
    }

    Tensor(TensorDimensions _dimensions, arma::Mat<T> _values)
            : D(std::move(_dimensions)),
              values(arma::field<arma::Mat<T>>(1)) {
        values.at(0, 0, 0) = _values;
    }

    Tensor(TensorDimensions _dimensions, arma::field<arma::Mat<T>> _values)
            : D(std::move(_dimensions)),
              values(_values) {}

    template<typename TNew>
    Tensor<TNew> ConvertTo() const {
        return Tensor<TNew>(
                D,
                values.for_each([](const arma::Mat<T> &e) {
                    return arma::conv_to<arma::Mat<TNew>>::from(e);
                })
        );
    }

    // note (sivukhin): used only in tests
    T &at(int x, int y) {
        return Values().at(x, y);
    }

    [[nodiscard]] arma::Mat<T> &Values() {
        ensure(Rank() <= 2);
        // note (sivukhin): .at(...) doesn't check bounds and all arma::field data just stored in single array
        return values.at(0, 0, 0);
    }

    [[nodiscard]] const arma::Mat<T> &Values() const {
        ensure(Rank() <= 2);
        // note (sivukhin): .at(...) doesn't check bounds and all arma::field data just stored in single array
        return values.at(0, 0, 0);
    }

    [[nodiscard]] const arma::field<arma::Mat<T>> &Field() const {
        return values;
    }

    [[nodiscard]] arma::field<arma::Mat<T>> &Field() {
        return values;
    }

    [[nodiscard]] int Rank() const {
        return D.size();
    }

    std::string ToString() const {
        auto result = std::string();
        for (size_t i = 0; i < D.size(); i++) {
            if (i != 0) {
                result += " x ";
            }
            result += std::to_string(D[i]);
        }
        std::stringstream stream;
        stream << "Tensor(" + std::string(typeid(T).name()) << ")" << "[" << result << "]" << std::endl
               << values << std::endl;
        return stream.str();
    }


    template<typename FillType>
    static Tensor<T> filled(TensorDimensions d, FillType fill) {
        static_assert(std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_ones>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_zeros>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randn>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randu>>::value != 0);
        auto values = createValuesContainer<T>(d);
        for (int a = 0; a < (d.size() >= 3 ? d[0] : 1); a++) {
            for (int b = 0; b < (d.size() >= 4 ? d[1] : 1); b++) {
                for (int c = 0; c < (d.size() >= 5 ? d[2] : 1); c++) {
                    values.at(a, b, c) = arma::Mat<T>(d[d.size() - 2], d.size() == 1 ? 1 : d.back(), fill);
                }
            }
        }
        return Tensor<T>(d, values);
    }

    static Tensor<T> init(TensorInitializer<T> initializer) {
        return Tensor<T>(initializer.D, initializer.values);
    }

    static Tensor<T> fromVector(const std::vector<std::vector<T>> &values) {
        auto tensor = Tensor<T>::filled({(int) values.size(), (int) values[0].size()}, arma::fill::zeros);
        for (size_t i = 0; i < values.size(); i++) {
            for (size_t s = 0; s < values[0].size(); s++) {
                tensor.Values().at(i, s) = values[i][s];
            }
        }
        return tensor;
    }

    TensorDimensions D;
private:
    arma::field<arma::Mat<T>> values;
};

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