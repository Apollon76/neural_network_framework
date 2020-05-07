#include "arma_math.hpp"

#include "utils.hpp"

arma::SizeMat Conv2dSize(const arma::SizeMat &matrix, const arma::SizeMat &, ConvolutionPadding padding) {
    if (padding == ConvolutionPadding::Same) {
        return matrix;
    } else {
        throw std::logic_error("Unsupported padding");
    }
}
