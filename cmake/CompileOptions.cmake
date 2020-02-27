set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# https://clang.llvm.org/docs/DiagnosticsReference.html
add_compile_options(-Wall -Wextra -Wpedantic -g -fno-omit-frame-pointer -Wshadow -Wno-unknown-pragmas -Wno-unused-result -Werror=return-type)

if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-O3)
endif()
