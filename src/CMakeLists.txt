cmake_minimum_required(VERSION 3.9)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package (JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIRS})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER protos/layers.proto)

file(GLOB PROTOBUF_DEFINITION_FILES "*.proto")
set(PROTOBUF_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
set(PROTOBUF_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src")
foreach(file ${PROTOBUF_DEFINITION_FILES})
    set(PROTOBUF_ARGUMENTS "protoc --proto_path=\"${PROTOBUF_INPUT_DIRECTORY}\" --cpp_out=\"${PROTOBUF_OUTPUT_DIRECTORY}\" \"${file}\"")
    execute_process(COMMAND ${PROTOBUF_OUTPUT_DIRECTORY}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE PROTOBUF_RESULT
            OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE)
endforeach()

file(GLOB PROTOBUF_MODELS_INCLUDES "src/*.pb.cc" "src/*.hpp")

add_library(neural_network_framework SHARED
        tensor.hpp
        layers/activations.hpp
        layers/dense.hpp
        layers/convolution2d.hpp
        optimizer.hpp
        neural_network.hpp
        utils.hpp
        loss.hpp
        data_processing/data_utils.hpp
        io/csv.hpp
        io/img.hpp
        io/filesystem.hpp
        scoring/scoring.hpp
        os_utils.hpp
        ${PROTO_HEADER}
        layers/layers_enum.hpp
        layers/layers_enum.cpp)

target_link_libraries(neural_network_framework ${ARMADILLO_LIBRARIES} pthread glog::glog ${PROTOBUF_LIBRARY} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} stdc++fs)
set_target_properties(neural_network_framework PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(neural_network_framework PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main neural_network_framework)
