cmake_minimum_required(VERSION 3.9)

add_executable(hdf5_example hd5_example.cpp)
target_link_libraries(hdf5_example neural_network_framework)
