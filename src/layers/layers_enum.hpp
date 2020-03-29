#pragma once

#include <string>

enum LayersEnum {
    DENSE,
    RELU_ACTIVATION,
    SOFTMAX_ACTIVATION,
    TANH_ACTIVATION,
    SIGMOID_ACTIVATION,
    CONV_2D,

    TOTAL
};

static std::string LayersName[LayersEnum::TOTAL] = {
        "Dense",
        "ReLUActivation",
        "SoftmaxActivation",
        "TanhActivation",
        "SigmoidActivation",
        "Conv2D"
};

std::string GetLayerNameByType(LayersEnum type);