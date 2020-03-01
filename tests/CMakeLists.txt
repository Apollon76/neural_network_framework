cmake_minimum_required(VERSION 3.9)

enable_testing()
find_package(GTest REQUIRED)

add_executable(unit_tests unit_tests.cpp)
target_link_libraries(unit_tests GTest::GTest neural_network_framework)