add_subdirectory(util)

add_executable(04_Exercise exercise_4_solution.cpp)
target_link_libraries(04_Exercise Kokkos::kokkos lightning_kokkos_utils)
target_include_directories(04_Exercise INTERFACE "util")
