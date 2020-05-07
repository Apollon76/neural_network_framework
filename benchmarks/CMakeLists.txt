cmake_minimum_required(VERSION 3.9)

add_executable(runner infrastructure/config_interface.hpp infrastructure/runner.cpp cases/mnist/config.hpp cases/cifar/config.hpp)
target_link_libraries(runner neural_network_framework)
