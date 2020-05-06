cmake_minimum_required(VERSION 3.9)

project(nn_framework)

include(cmake/Logging.cmake)
include(cmake/CheckCompiler.cmake)
include(cmake/CompileOptions.cmake)
include(cmake/Sanitize.cmake)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
