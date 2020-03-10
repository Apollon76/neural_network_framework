cmake_minimum_required(VERSION 3.9)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER protos/layers.proto)

add_executable(benchmark dense_benchmark.cpp)
target_link_libraries(benchmark neural_network_framework ${PROTOBUF_LIBRARY} proto)