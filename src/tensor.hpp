#pragma once

#include <armadillo>
#include <utility>
#include <vector>
#include <src/utils.hpp>

using TensorDimensions = std::vector<int>;

template<typename T>
arma::field<arma::Mat<T>> createValuesContainer(TensorDimensions d) {
    ensure(d.size() <= 5, "Supported only tensors with not more than 5 dimension");
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
class TensorInitializer {
public:
    TensorInitializer(T x) // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
            : D(),
              values([x]() {
                  auto field = createValuesContainer<T>({1});
                  field.at(0) = arma::Mat<T>({x});
                  return field;
              }()) {
    }

    TensorInitializer(std::initializer_list<TensorInitializer<T>> list) : D(), values() {
        auto other = std::optional<TensorDimensions>();
        auto valueList = std::vector<arma::field<arma::Mat<T>>>();
        for (auto &&item : list) {
            ensure(!other.has_value() || item.D == other.value(),
                   "Nested tensors must have equal dimensions");
            other = item.D;
            valueList.push_back(item.values);
        }
        ensure(other.has_value(), "Tensor dimensions must be non-empty");
        D = other.value();
        D.insert(D.begin(), list.size());
        values = createValuesContainer<T>(D);
        if (D.size() <= 2) {
            auto matrix = arma::Mat<T>(D.size() > 0 ? D[0] : 1, D.size() > 1 ? D[1] : 1);
            for (size_t i = 0; i < matrix.n_rows; i++) {
                for (size_t s = 0; s < matrix.n_cols; s++) {
                    matrix.at(i, s) = valueList[i].at(0, 0, 0).at(s);
                }
            }
            values.at(0, 0, 0) = matrix;
        } else {
            for (int a = 0; a < (D.size() >= 3 ? D[0] : 1); a++) {
                for (int b = 0; b < (D.size() >= 4 ? D[1] : 1); b++) {
                    for (int c = 0; c < (D.size() >= 5 ? D[2] : 1); c++) {
                        values.at(a, b, c) = valueList[a].at(b, c);
                    }
                }
            }
        }
    }

    TensorDimensions D;
    arma::field<arma::Mat<T>> values;
};

template<typename T>
class Tensor {
public:
    Tensor()
            : D(),
              values(arma::field<arma::Mat<T>>(1)) {
    }

    Tensor(TensorDimensions _dimensions, arma::Mat<T> _values)
            : D(std::move(_dimensions))
            , values(arma::field<arma::Mat<T>>(1))
            , saved_mat(true)
    {
        values.at(0, 0, 0) = _values;
    }

    Tensor(TensorDimensions _dimensions, arma::field<arma::Mat<T>> _values)
            : D(std::move(_dimensions)),
              values(_values) {}


    void ForEach(const std::function<void(int, int, int, arma::Mat<T> &)> &f) {
        for (int a = 0; a < (D.size() >= 3 ? D[0] : 1); a++) {
            for (int b = 0; b < (D.size() >= 4 ? D[1] : 1); b++) {
                for (int c = 0; c < (D.size() >= 5 ? D[2] : 1); c++) {
                    f(a, b, c, values.at(a, b, c));
                }
            }
        }
    }

    void ForEach(const std::function<void(int, int, int, const arma::Mat<T> &)> &f) const {
        for (int a = 0; a < (D.size() >= 3 ? D[0] : 1); a++) {
            for (int b = 0; b < (D.size() >= 4 ? D[1] : 1); b++) {
                for (int c = 0; c < (D.size() >= 5 ? D[2] : 1); c++) {
                    f(a, b, c, values.at(a, b, c));
                }
            }
        }
    }

    template<typename TNew>
    Tensor<TNew> Transform(const std::function<arma::Mat<TNew>(const arma::Mat<T> &)> &f) const {
        auto newValues = Tensor<TNew>(D, createValuesContainer<TNew>(D));
        newValues.ForEach([&f, this](int a, int b, int c, arma::Mat<TNew> &value) {
            value = f(values.at(a, b, c));
        });
        return newValues;
    }

    template<typename TNew>
    TNew Aggregate(TNew initial, const std::function<void(TNew &, const arma::Mat<T> &)> &f) const {
        ForEach([&initial, &f](int, int, int, const arma::Mat<T> &v) {
            f(initial, v);
        });
        return initial;
    }

    template<typename TNew>
    Tensor<TNew> DiffWith(
            const Tensor<T> &other,
            const std::function<arma::Mat<TNew>(const arma::Mat<T> &, const arma::Mat<T> &)> &f
    ) const {
        ensure(D == other.D, "dimensions must be equal");
        auto newValues = Tensor<TNew>(D, createValuesContainer<TNew>(D));
        newValues.ForEach([&f, this, &other](int a, int b, int c, arma::Mat<T> &v) {
            v = f(values.at(a, b, c), other.values.at(a, b, c));
        });
        // todo (sivukhin): fix dimension here
        return newValues;
    }

    template<typename TNew>
    Tensor<TNew> ConvertTo() const {
        return Transform<TNew>([](const arma::Mat<T> &e) {
            return arma::conv_to<arma::Mat<TNew>>::from(e);
        });
    }

    // note (sivukhin): used only in tests
    T &at(int x, int y) {
        return Values().at(x, y);
    }

    [[nodiscard]] arma::Mat<T> &Values() {
        ensure(Rank() <= 2, "Rank of tensor must be not more than 2 for Values extraction");
        // note (sivukhin): .at(...) doesn't check bounds and all arma::field data just stored in single array
        return values.at(0, 0, 0);
    }

    [[nodiscard]] const arma::Mat<T> &Values() const {
        ensure(Rank() <= 2, "Rank of tensor must be not more than 2 for Values extraction");
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
        std::stringstream stream;
        stream << "Tensor(" + std::string(typeid(T).name()) << ")" << "[" << FormatDimensions(D) << "]" << std::endl
               << values << std::endl;
        return stream.str();
    }


    template<typename FillType>
    static Tensor<T> filled(TensorDimensions d, FillType fill) {
        static_assert(std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_ones>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_zeros>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randn>>::value != 0 ||
                      std::is_same<FillType, arma::fill::fill_class<arma::fill::fill_randu>>::value != 0);
        auto emptyTensor = Tensor<T>(d, createValuesContainer<T>(d));
        return emptyTensor.template Transform<T>(
                [&d, &fill](const arma::Mat<T> &) {
                    if (d.size() == 0) {
                        return arma::Mat<T>();
                    } else if (d.size() == 1) {
                        return arma::Mat<T>(d[0], 1, fill);
                    } else {
                        return arma::Mat<T>(d[d.size() - 2], d[d.size() - 1], fill);
                    }
                });
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

    Tensor<T> Rows(const arma::uvec& rows) const {
        auto dim = D;
        dim[0] = rows.size();
        if (saved_mat) {
            return Tensor<T>(dim, values.at(0, 0, 0).rows(rows));
        }
        auto new_data = Tensor<T>(dim, createValuesContainer<T>(dim));
        if (Rank() <= 2) {
            new_data.values(0) = values(0).rows(rows);
        } else {
            for (size_t i = 0; i < rows.size(); ++i) {
                new_data.values.row(i) = values.row(rows[i]);
            }
        }
        return new_data;
    }

    TensorDimensions D;
private:
    arma::field<arma::Mat<T>> values;
    bool saved_mat = false;
};

template<typename T>
std::string FormatDimensions(const Tensor<T> &t) {
    return FormatDimensions(t.D);
}