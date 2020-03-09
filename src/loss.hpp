#pragma once

#include <armadillo>


class ILoss : public ISerializable {
public:
    [[nodiscard]] virtual double GetLoss(const arma::mat &inputs, const arma::mat &outputs) const = 0;

    [[nodiscard]] virtual arma::mat GetGradients(const arma::mat &input, const arma::mat &outputs) const = 0;

    virtual ~ILoss() = default;
};

class MSELoss : public ILoss {
public:
    [[nodiscard]] double GetLoss(const arma::mat &inputs, const arma::mat &outputs) const override {
        return arma::accu(arma::pow(inputs - outputs, 2)) / inputs.n_rows;
    }

    [[nodiscard]] arma::mat GetGradients(const arma::mat &inputs, const arma::mat &outputs) const override {
        return 2 * (inputs - outputs) / inputs.n_rows;
    }

    [[nodiscard]] json Serialize() const override {
        return {"loss_type", "mse"};
    }
};

class CategoricalCrossEntropyLoss : public ILoss {
 public:
    [[nodiscard]] double GetLoss(const arma::mat &inputs, const arma::mat &outputs) const override {
        return arma::mean(arma::sum(outputs % arma::log(inputs), 1) * -1);
    }

    [[nodiscard]] arma::mat GetGradients(const arma::mat &inputs, const arma::mat &outputs) const override {
        return -outputs / inputs;
    }

    [[nodiscard]] json Serialize() const override {
        return {"loss_type", "categorical_cross_entropy"};
    }
};