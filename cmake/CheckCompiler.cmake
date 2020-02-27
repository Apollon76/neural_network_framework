set(REQUIRED_CXX_COMPILER "Clang")
set(CXX_COMPILER_MIN_VERSION 6.0)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL REQUIRED_CXX_COMPILER)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS CXX_COMPILER_MIN_VERSION)
        log_fatal(
            "Old version of ${REQUIRED_CXX_COMPILER} compiler: ${CMAKE_CXX_COMPILER_VERSION}, required ${CXX_COMPILER_MIN_VERSION}."
            )
    endif()
else()
    log_fatal(
        "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. Use ${REQUIRED_CXX_COMPILER}, version >= ${CXX_COMPILER_MIN_VERSION}."
        )
endif()
