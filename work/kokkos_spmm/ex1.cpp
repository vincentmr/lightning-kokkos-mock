#include "Kokkos_Core.hpp"
#include <cmath>

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define MemSpace Kokkos::Experimental::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#define MemSpace Kokkos::OpenMPTargetSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using Layout = Kokkos::LayoutRight;

using Scalar = Kokkos::complex<double>;
using Ordinal = default_lno_t;
using Offset = default_size_type;

using device_type = typename Kokkos::Device<ExecSpace, MemSpace>;
using matrix_type = typename KokkosSparse::CrsMatrix<Scalar, Ordinal,
                                                     device_type, void, Offset>;
using graph_type = typename matrix_type::staticcrsgraph_type;
using row_map_type = typename graph_type::row_map_type;
using entries_type = typename graph_type::entries_type;
using values_type = typename matrix_type::values_type;

matrix_type get_sparse_matrix(Ordinal sblk);
void set_sparse_data(Ordinal sblk, row_map_type::HostMirror &row_map_h,
                     entries_type::HostMirror &entries_h,
                     values_type::HostMirror &values_h);

int main(int argc, char *argv[]) {
  Kokkos::initialize();

  {
    const Scalar SC_ONE = Kokkos::ArithTraits<Scalar>::one();

    Ordinal T = 10;
    Ordinal numRows = std::pow(2, T + 1);
    Ordinal numCols = std::pow(2, 25 - 1 - T);
    Ordinal S = numRows * numCols;
    Ordinal nrepeat = 10;

    // Build the row pointers and store numNNZ
    typename row_map_type::non_const_type row_map("row pointers", numRows + 1);
    typename row_map_type::HostMirror row_map_h =
        Kokkos::create_mirror_view(row_map);
    const Offset numNNZ = numRows * 2;
    Kokkos::deep_copy(row_map, row_map_h);

    typename entries_type::non_const_type entries("column indices", numNNZ);
    typename entries_type::HostMirror entries_h =
        Kokkos::create_mirror_view(entries);
    typename values_type::non_const_type values("values", numNNZ);
    typename values_type::HostMirror values_h =
        Kokkos::create_mirror_view(values);

    set_sparse_data(numRows / 2, row_map_h, entries_h, values_h);

    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);

    graph_type myGraph(entries, row_map);
    matrix_type myMatrix("test matrix", numRows, values, myGraph);

    const Scalar alpha = SC_ONE;
    const Scalar beta = SC_ONE;

    Kokkos::View<Scalar **, Layout, MemSpace> x("lhs", numRows, numCols);
    Kokkos::View<Scalar **, Layout, MemSpace> y("rhs", numRows, numCols);

    Kokkos::fence();
    Kokkos::Timer timer;
    for (int i = 0; i < nrepeat; i++) {
      KokkosSparse::spmv("N", alpha, myMatrix, x, beta, y);
    }
    Kokkos::fence();

    // Calculate time.
    double time = timer.seconds();

    // Calculate bandwidth.
    double Gbytes = 1.0e-9 * double(sizeof(Scalar) * S);

    // Print results (problem size, time and bandwidth in GB/s).
    printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
           time, Gbytes * nrepeat / time);
  }

  Kokkos::finalize();

  return 0;
}

void set_sparse_data(Ordinal sblk, row_map_type::HostMirror &row_map_h,
                     entries_type::HostMirror &entries_h,
                     values_type::HostMirror &values_h) {
  std::size_t count = 0;
  std::size_t i;

  for (i = 0; i < sblk; i++) {
    row_map_h(i) = count;
    entries_h(count) = i;
    values_h(count) = static_cast<Scalar>(1.0);
    count += 1;
    entries_h(count) = i + sblk;
    values_h(count) = static_cast<Scalar>(1.0);
    count += 1;
  }
  for (i = sblk; i < 2 * sblk; i++) {
    row_map_h(i) = count;
    entries_h(count) = i - sblk;
    values_h(count) = static_cast<Scalar>(1.0);
    count += 1;
    entries_h(count) = i;
    values_h(count) = static_cast<Scalar>(1.0);
    count += 1;
  }
}
