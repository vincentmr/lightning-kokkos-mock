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
find_package(KokkosKernels REQUIRED)
add_library(kokkos ALIAS Kokkos::kokkos)
add_library(kokkoskernels ALIAS Kokkos::kokkoskernels)

add_subdirectory(src)

