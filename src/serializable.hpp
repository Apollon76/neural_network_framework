#pragma once

#include <nlohmann/json.hpp>

using json = nlohmann::json;


class ISerializable {
public:
    virtual json Serialize() const = 0;

    virtual void FromJson(json data) = 0;
};
