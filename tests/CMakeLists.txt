cmake_minimum_required(VERSION 3.9)

enable_testing()
find_package(GTest REQUIRED)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER protos/layers.proto)
add_library(proto ${PROTO_HEADER} ${PROTO_SRC})

add_executable(unit_tests entry_point.cpp unit_tests.cpp test_scoring.cpp test_serialization.cpp utils.cpp)
target_link_libraries(unit_tests GTest::GTest neural_network_framework ${PROTOBUF_LIBRARY} proto)