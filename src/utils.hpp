#pragma once

#include <vector>
#include <chrono>
#include <glog/logging.h>


#define UNUSED(x)   \
do {                \
    (void) x;      \
} while(false);


constexpr void ensure(bool value, const std::string &error) {
    if (!value) {
        throw std::runtime_error(error);
    }
}

constexpr void ensure(bool value) {
    if (!value) {
        throw std::runtime_error("precondition check failed");
    }
}

template<typename T>
std::string FormatDimensions(const arma::Mat<T> &mat) {
    return std::to_string(mat.n_rows) + "x" + std::to_string(mat.n_cols);
}

template<typename T>
std::string FormatDimensions(const arma::Cube<T> &cube) {
    return std::to_string(cube.n_rows) + "x" + std::to_string(cube.n_cols) + "x" + std::to_string(cube.n_slices);
}

class Timer {
public:
    explicit Timer(std::string action_description, bool use_stdout = false) :
            begin(std::chrono::steady_clock::now()),
            action_description(std::move(action_description)),
            use_stdout(use_stdout) {
    }

    Timer(const Timer &) = delete;

    Timer(const Timer &&) = delete;

    Timer &operator=(const Timer &) = delete;

    Timer &operator=(const Timer &&) = delete;

    ~Timer() {
        auto end = std::chrono::steady_clock::now();
        (use_stdout ? std::cout : LOG(INFO))
                << action_description << " done in: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                << " ms" << std::endl;

    }

private:
    decltype(std::chrono::steady_clock::now()) begin;
    std::string action_description;
    bool use_stdout;
};