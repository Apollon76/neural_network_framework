#pragma once

#include <armadillo>
#include <algorithm>
#include "utils.hpp"

enum ConvolutionPadding {
    Same,
};

arma::SizeMat Conv2dSize(const arma::SizeMat &matrix, const arma::SizeMat &kernel, ConvolutionPadding padding);

// note (sivukhin): Простая реализация конволюции. Решил написать свою по слеудющим причинам:
// 1. arma::conv2 реализует не совсем понятную конволюцию (матрица ядра как-то поворачивается, транспорнируется) и велика вероятность запутаться
// 2. Наивная реализация в этом смысле проще для понимания, а быстрые эксперименты показывают замедление на 30%, что не супер круто, но и не сильно страшно
// 3. В документации написано, что реализация сырая и неоптимизированная, поэтому возможно и так придется что-то самим выдумывать
template<typename T>
arma::Mat<T> Conv2d(const arma::Mat<T> &matrix, const arma::Mat<T> &kernel, ConvolutionPadding padding) {
    auto result = arma::Mat<T>(Conv2dSize(arma::size(matrix), arma::size(kernel), padding), arma::fill::zeros);
    for (arma::uword k_x = 0; k_x < kernel.n_rows; k_x++) {
        auto row_multiplier = arma::Mat<T>(matrix.n_cols, matrix.n_cols, arma::fill::zeros);
        for (arma::uword y = 0; y < matrix.n_cols; y++) {
            for (arma::uword k_y = 0; y + k_y < matrix.n_cols && k_y < kernel.n_cols; k_y++) {
                row_multiplier(y + k_y, y) = kernel(k_x, k_y);
            }
        }
        arma::Mat<T> row_result = matrix * row_multiplier;
        result.rows(0, matrix.n_rows - k_x - 1) += row_result.rows(k_x, matrix.n_rows - 1);
    }
    return result;
}

template<typename T>
arma::Mat<T> Mirror(const arma::Mat<T> &matrix) {
    return arma::fliplr(arma::flipud(matrix));
}