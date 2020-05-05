#include "arma_math.hpp"

#include "utils.hpp"

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
