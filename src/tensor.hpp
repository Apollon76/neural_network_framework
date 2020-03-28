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
        return arma::field<arma::Mat<T>>(1, 1, 1);
    } else if (d.size() == 3) {
        return arma::field<arma::Mat<T>>(1, 1, d[0]);
    } else if (d.size() == 4) {
        return arma::field<arma::Mat<T>>(1, d[0], d[1]);
    } else if (d.size() == 5) {
        return arma::field<arma::Mat<T>>(d[0], d[1], d[2]);
    } else {
        throw std::logic_error("too many dimensions");
    }
}

template<typename T>
class Tensor;

template<typename T>
class TensorConstView {
public:
    TensorConstView(const Tensor<T> &_ref, std::vector<int> _indices);

    TensorConstView<T> View(int id) const;

    TensorConstView<T> View(int x, int y) const;

    const arma::Mat<T> &Matrix() const;

    const T &At(int x, int y) const;

private:
    const Tensor<T> &ref;
    int fixed;
    std::array<int, 3> indices;
};

template<typename T>
class TensorView {
public:
    TensorView(Tensor<T> &_ref, TensorDimensions _indices);

    TensorView<T> View(int id) const;

    TensorView<T> View(int x, int y) const;

    const arma::Mat<T> &Matrix() const;

    arma::Mat<T> &Matrix();

    const T &At(int x, int y) const;

    T &At(int x, int y);

private:
    Tensor<T> &ref;
    int fixed;
    std::array<size_t, 3> indices;
};

template<typename T>
class Tensor {
    friend TensorView<T>;
    friend TensorConstView<T>;

public:
    Tensor()
            : D(),
              values(arma::field<arma::Mat<T>>(1, 1, 1)) {
    }

    Tensor(TensorDimensions _dimensions, arma::Mat<T> _values)
            : D(std::move(_dimensions)),
              values(arma::field<arma::Mat<T>>(1, 1, 1)) {
        values.at(0, 0, 0) = _values;
    }

    Tensor(TensorDimensions _dimensions, arma::field<arma::Mat<T>> _values)
            : D(std::move(_dimensions)),
              values(_values) {}

    template<typename TNew>
    Tensor<TNew> ConvertTo() const {
        auto result = arma::field<arma::Mat<TNew>>(values.n_rows, values.n_cols, values.n_slices);
        for (int i = 0; i < (int) values.n_rows; i++) {
            for (int s = 0; s < (int) values.n_cols; s++) {
                for (int k = 0; k < (int) values.n_slices; k++) {
                    result.at(i, s, k) = arma::conv_to<arma::Mat<TNew>>::from(values.at(i, s, k));
                }
            }
        }
        return Tensor<TNew>(D, result);
    }

    const T &at(int x, int y) const {
        return TensorConstView<T>(*this, {}).At(x, y);
    }

    TensorView<T> View() {
        return TensorView<T>(*this, {});
    }

    TensorConstView<T> ConstView() const {
        return TensorConstView<T>(*this, {});
    }

    T &at(int x, int y) {
        return TensorView<T>(*this, {}).At(x, y);
    }

    [[nodiscard]] arma::Mat<T> &Values() {
        return TensorView<T>(*this, {}).Matrix();
    }

    [[nodiscard]] const arma::Mat<T> &Values() const {
        return TensorConstView<T>(*this, {}).Matrix();
    }

    [[nodiscard]] int Rank() const {
        return D.size();
    }

    [[nodiscard]] int BatchCount() const {
        return D[0];
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
        for (int a = 0; a < (d.size() >= 5 ? d[d.size() - 5] : 1); a++) {
            for (int b = 0; b < (d.size() >= 4 ? d[d.size() - 4] : 1); b++) {
                for (int c = 0; c < (d.size() >= 3 ? d[d.size() - 3] : 1); c++) {
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
TensorConstView<T>::TensorConstView(const Tensor<T> &_ref, std::vector<int> _indices)
        : ref(_ref), fixed(_indices.size()), indices() {
    ensure(_indices.size() <= 3);
    int shift = 3 - (int) _indices.size();
    for (int i = 0; i < 3; i++) {
        indices[i] = i >= shift ? _indices[i - shift] : 0;
    }
}

template<typename T>
TensorConstView<T> TensorConstView<T>::View(int id) const {
    ensure(fixed + 1 <= ref.Rank());
    auto new_indices = indices;
    new_indices.push_back(id);
    return TensorConstView<T>(ref, new_indices);
}

template<typename T>
TensorConstView<T> TensorConstView<T>::View(int x, int y) const {
    ensure(fixed + 2 <= ref.Rank());
    auto new_indices = TensorDimensions();
    for (int i = 3 - fixed; i < 3; i++) {
        new_indices.push_back(indices[i]);
    }
    new_indices.push_back(x);
    new_indices.push_back(y);
    return TensorConstView<T>(ref, new_indices);
}

template<typename T>
const arma::Mat<T> &TensorConstView<T>::Matrix() const {
    ensure(fixed + 2 >= ref.Rank() && fixed <= ref.Rank());
    return ref.values.at(indices[0], indices[1], indices[2]);
}

template<typename T>
const T &TensorConstView<T>::At(int x, int y) const {
    return Matrix().at(x, y);
}

template<typename T>
TensorView<T>::TensorView(Tensor<T> &_ref, TensorDimensions _indices) : ref(_ref), fixed(_indices.size()), indices() {
    ensure(_indices.size() <= 3);
    int shift = 3 - (int) _indices.size();
    for (int i = 0; i < 3; i++) {
        indices[i] = i >= shift ? _indices[i - shift] : 0;
    }
}

template<typename T>
TensorView<T> TensorView<T>::View(int id) const {
    ensure(fixed + 1 <= ref.Rank());
    auto new_indices = indices;
    new_indices.push_back(id);
    return TensorView<T>(ref, new_indices);
}

template<typename T>
TensorView<T> TensorView<T>::View(int x, int y) const {
    ensure(fixed + 2 <= ref.Rank());
    auto new_indices = TensorDimensions();
    for (int i = 3 - fixed; i < 3; i++) {
        new_indices.push_back(indices[i]);
    }
    new_indices.push_back(x);
    new_indices.push_back(y);
    return TensorView<T>(ref, new_indices);
}

template<typename T>
arma::Mat<T> &TensorView<T>::Matrix() {
    ensure(fixed + 2 == ref.Rank() || fixed + 1 == ref.Rank());
    return ref.values.at(indices[0], indices[1], indices[2]);
}

template<typename T>
const arma::Mat<T> &TensorView<T>::Matrix() const {
    ensure(fixed + 2 == ref.Rank() || fixed + 1 == ref.Rank());
    return ref.values.at(indices[0], indices[1], indices[2]);
}

template<typename T>
T &TensorView<T>::At(int x, int y) {
    return Matrix().at(x, y);
}

template<typename T>
const T &TensorView<T>::At(int x, int y) const {
    return Matrix().at(x, y);
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