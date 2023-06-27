cmake_minimum_required (VERSION 3.20)
project (Lightning-Kokkos
    DESCRIPTION "C++ state-vector simulator bindings for PennyLane. "
    LANGUAGES CXX
)

set(PROJECT_VERSION "1.0.0")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Kokkos REQUIRED)
add_library(kokkos ALIAS Kokkos::kokkos)

add_subdirectory(src)

