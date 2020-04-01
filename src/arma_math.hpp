#pragma once

#include <armadillo>
#include <algorithm>

enum ConvolutionPadding {
    Same,
    Valid,
};

arma::SizeMat Conv2dSize(const arma::SizeMat &matrix, const arma::SizeMat &kernel, ConvolutionPadding padding) {
    if (padding == ConvolutionPadding::Same) {
        return matrix;
    } else if (padding == ConvolutionPadding::Valid) {
        ensure(kernel.n_rows <= matrix.n_rows && kernel.n_cols <= matrix.n_cols,
               "Kernel must be smaller than matrix");
        return arma::SizeMat(matrix.n_rows - kernel.n_rows + 1, matrix.n_cols - kernel.n_cols + 1);
    } else {
        throw std::logic_error("Unsupported padding");
    }
}

// note (sivukhin): Простая реализация конволюции. Решил написать свою по слеудющим причинам:
// 1. arma::conv2 реализует не совсем понятную конволюцию (матрица ядра как-то поворачивается, транспорнируется) и велика вероятность запутаться
// 2. Наивная реализация в этом смысле проще для понимания, а быстрые эксперименты показывают замедление на 30%, что не супер круто, но и не сильно страшно
// 3. В документации написано, что реализация сырая и неоптимизированная, поэтому возможно и так придется что-то самим выдумывать
template<typename T>
arma::Mat<T> Conv2d(const arma::Mat<T> &matrix, const arma::Mat<T> &kernel, ConvolutionPadding padding) {
    auto result = arma::Mat<T>(Conv2dSize(arma::size(matrix), arma::size(kernel), padding), arma::fill::zeros);
    if (padding == ConvolutionPadding::Valid) {
        for (arma::uword i = 0; i < result.n_rows; i++) {
            for (arma::uword s = 0; s < result.n_cols; s++) {
                auto subX = matrix.submat(i, s, i + kernel.n_rows - 1, s + kernel.n_cols - 1);
                result(i, s) = arma::accu(subX % kernel);
            }
        }
    } else {
        int left_padding = kernel.n_cols / 2;
        int up_padding = kernel.n_rows / 2;

        for (int i = -up_padding; i < -up_padding + int64_t(matrix.n_rows); ++i) {
            for (int s = -left_padding; s < -left_padding + int64_t(matrix.n_cols); ++s) {
                auto subX = matrix.submat(
                        std::max<int>(i, 0),
                        std::max<int>(s, 0),
                        std::min<int>(i + kernel.n_rows - 1, matrix.n_rows - 1),
                        std::min<int>(s + kernel.n_cols - 1, matrix.n_cols - 1)
                );
                auto subY = kernel.submat(
                        -std::min(i, 0),
                        -std::min(s, 0),
                        -std::min(i, 0) + subX.n_rows - 1,
                        -std::min(s, 0) + subX.n_cols - 1
                );
                result(i + up_padding, s + left_padding) = arma::accu(subX % subY);
            }
        }
    }
    return result;
}

template<typename T>
arma::Mat<T> Mirror(const arma::Mat<T> &matrix) {
    return arma::fliplr(arma::flipud(matrix));
}