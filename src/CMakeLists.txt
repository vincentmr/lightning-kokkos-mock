add_subdirectory(gates)
add_subdirectory(util)

add_executable(lightning_kokkos lightning_kokkos.cpp)
target_link_libraries(lightning_kokkos Kokkos::kokkos lightning_kokkos_gates lightning_kokkos_utils)
target_include_directories(lightning_kokkos INTERFACE "gates" "util")
