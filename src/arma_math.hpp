#pragma once

#include <armadillo>
#include <algorithm>
#include "utils.hpp"

enum ConvolutionPadding {
    Same,
    Valid,
};

arma::SizeMat Conv2dSize(const arma::SizeMat &matrix, const arma::SizeMat &kernel, ConvolutionPadding padding);

// note (sivukhin): Простая реализация конволюции. Решил написать свою по слеудющим причинам:
// 1. arma::conv2 реализует не совсем понятную конволюцию (матрица ядра как-то поворачивается, транспорнируется) и велика вероятность запутаться
// 2. Наивная реализация в этом смысле проще для понимания, а быстрые эксперименты показывают замедление на 30%, что не супер круто, но и не сильно страшно
// 3. В документации написано, что реализация сырая и неоптимизированная, поэтому возможно и так придется что-то самим выдумывать
template<typename T>
arma::Mat<T> Conv2d(const arma::Mat<T> &matrix, const arma::Mat<T> &kernel, ConvolutionPadding padding) {
    auto result = arma::Mat<T>(Conv2dSize(arma::size(matrix), arma::size(kernel), padding), arma::fill::zeros);
    for (arma::uword i = 0; i < result.n_rows; i++) {
        for (arma::uword s = 0; s < result.n_cols; s++) {
            auto subX = matrix.submat(
                    i,
                    s,
                    std::min(i + kernel.n_rows - 1, matrix.n_rows - 1),
                    std::min(s + kernel.n_cols - 1, matrix.n_cols - 1)
            );
            auto subY = kernel.submat(
                    0,
                    0,
                    std::min(kernel.n_rows - 1, matrix.n_rows - i - 1),
                    std::min(kernel.n_cols - 1, matrix.n_cols - s - 1)
            );
            result(i, s) = arma::accu(subX % subY);
        }
    }
    return result;
}

template<typename T>
arma::Mat<T> Mirror(const arma::Mat<T> &matrix) {
    return arma::fliplr(arma::flipud(matrix));
}