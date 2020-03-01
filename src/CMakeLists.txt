cmake_minimum_required(VERSION 3.9)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

add_executable(main main.cpp layers.hpp optimizer.hpp neural_network.hpp)

target_link_libraries(main ${ARMADILLO_LIBRARIES} pthread glog::glog)
