cmake_minimum_required(VERSION 3.9)

add_executable(runner infrastructure/config_interface.cpp infrastructure/runner.cpp cases/mnist/config.cpp)
target_link_libraries(runner neural_network_framework)
