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

add_executable(ex1 ex1.cpp)
target_link_libraries(ex1 Kokkos::kokkos Kokkos::kokkoskernels)

