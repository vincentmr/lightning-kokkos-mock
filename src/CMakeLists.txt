add_subdirectory(gates)
# add_subdirectory(util)

add_executable(lightning_kokkos lightning_kokkos.cpp)
target_link_libraries(lightning_kokkos kokkos kokkoskernels lightning_kokkos_gates)
