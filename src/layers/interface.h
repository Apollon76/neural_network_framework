#pragma once

#include <armadillo>
#include <vector>
#include <glog/logging.h>


struct Gradients {
    arma::mat input_gradients;
    arma::mat layer_gradients;
};

class ILayer {
public:
    [[nodiscard]] virtual std::string ToString() const = 0;

    [[nodiscard]] virtual std::string GetName() const = 0;

    [[nodiscard]] virtual arma::mat Apply(const arma::mat &) const = 0;

    [[nodiscard]] virtual Gradients PullGradientsBackward(
            const arma::mat &input,
            const arma::mat &output_gradients
    ) const = 0;

    virtual void ApplyGradients(const arma::mat &gradients) = 0;

    virtual ~ILayer() = default;
};