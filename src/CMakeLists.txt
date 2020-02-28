cmake_minimum_required(VERSION 3.9)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main ${ARMADILLO_LIBRARIES})
target_link_libraries(main pthread)
target_link_libraries(main gfortran)
