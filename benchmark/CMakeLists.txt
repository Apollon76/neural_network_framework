cmake_minimum_required(VERSION 3.9)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(benchmark dense_benchmark.cpp)
target_link_libraries(benchmark neural_network_framework)