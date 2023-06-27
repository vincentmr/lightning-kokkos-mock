project(lightning_kokkos_gates LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(lightning_kokkos_gates INTERFACE)
target_include_directories(lightning_kokkos_gates INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DS>
)
