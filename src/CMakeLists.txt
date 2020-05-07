cmake_minimum_required(VERSION 3.9)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package (JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIRS})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

find_package(HighFive REQUIRED)

find_package(HDF5 REQUIRED)

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
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
        layers/dropout.hpp
        layers/convolution2d.hpp
        layers/flatten.hpp
        layers/max_pooling2d.hpp
        optimizer.hpp
        neural_network.hpp
        neural_network_interface.hpp
        utils.hpp
        loss.hpp
        data_processing/data_utils.hpp
        io/csv.hpp
        io/img.hpp
        io/filesystem.hpp
        serialization/hdf5_serialization.hpp
        scoring/scoring.hpp
        arma_math.hpp
        arma_math.cpp
        callbacks/interface.hpp
        callbacks/meta_callbacks.hpp
        callbacks/performance_metrics_callback.hpp
        callbacks/logging_callback.hpp
        callbacks/progress_bar_callback.hpp
        ${PROTO_HEADER}
        ${PROTO_SRC})

target_link_libraries(neural_network_framework
        ${ARMADILLO_LIBRARIES}
        ${OpenMP_LIBS}
        pthread
        glog::glog
        ${PROTOBUF_LIBRARY}
        ${PNG_LIBRARIES}
        ${JPEG_LIBRARIES}
        stdc++fs
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
        ${HDF5_LIBRARY_DIRS})

target_include_directories(neural_network_framework PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR} # It's required for protobuf for some reason...
        ${HDF5_INCLUDE_DIRS})

set_target_properties(neural_network_framework PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main main.cpp)
target_link_libraries(main neural_network_framework)
