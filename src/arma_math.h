#pragma once

#include <armadillo>
#include <algorithm>

enum ConvolutionPadding {
    Same,
    Valid,
};

arma::SizeMat Conv2dSize(const arma::SizeMat &matrix, const arma::SizeMat &kernel, ConvolutionPadding padding);

template<typename T>
arma::Mat<T> Conv2d(const arma::Mat<T> &matrix, const arma::Mat<T> &kernel, ConvolutionPadding padding);

template<typename T>
arma::Mat<T> Mirror(const arma::Mat<T> &matrix);