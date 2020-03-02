cmake_minimum_required(VERSION 3.9)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

add_library(neural_network_framework SHARED layers.hpp optimizer.hpp neural_network.hpp utils.hpp loss.hpp)
target_link_libraries(neural_network_framework ${ARMADILLO_LIBRARIES} pthread glog::glog)
set_target_properties(neural_network_framework PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(neural_network_framework PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main neural_network_framework)