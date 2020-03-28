#pragma once

#include <armadillo>
#include <cereal/types/polymorphic.hpp>


class ILoss {
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

    template<class Archive>
    void serialize(Archive& ar) {
        ar();
    }
};

CEREAL_REGISTER_TYPE(MSELoss)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILoss, MSELoss)

class CategoricalCrossEntropyLoss : public ILoss {
 public:
    [[nodiscard]] double GetLoss(const arma::mat &inputs, const arma::mat &outputs) const override {
        return arma::mean(arma::sum(outputs % arma::log(inputs), 1) * -1);
    }

    [[nodiscard]] arma::mat GetGradients(const arma::mat &inputs, const arma::mat &outputs) const override {
        return -outputs / inputs;
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar();
    }
};

CEREAL_REGISTER_TYPE(CategoricalCrossEntropyLoss)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ILoss, CategoricalCrossEntropyLoss)