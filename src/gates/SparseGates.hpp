#pragma once

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/time.h>
#include <vector>

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "Kokkos_Core.hpp"

#include "Gates.hpp"

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
using layout = Kokkos::LayoutLeft;

using device_type = typename Kokkos::Device<ExecSpace, MemSpace>;

using index_type = long int;
using index_view_type = Kokkos::View<index_type *, layout, MemSpace>;

using data_type = vectorType;
// using data_type = Kokkos::complex<double>;

using crs_matrix_type =
    typename KokkosSparse::CrsMatrix<data_type, index_type, device_type, void,
                                     index_type>;

using graph_type = typename crs_matrix_type::staticcrsgraph_type;

using data_view_type = Kokkos::View<vectorType *, layout, MemSpace>;

crs_matrix_type create_Kokkos_Sparse_Matrix(index_type *row_map_ptr,
                                            const index_type numRows,
                                            index_type *entries_ptr,
                                            data_type *values_ptr,
                                            const index_type numNNZ) {
  index_view_type row_map(row_map_ptr, numRows + 1);
  index_view_type entries(entries_ptr, numNNZ);
  data_view_type values(values_ptr, numNNZ);

  graph_type myGraph(entries, row_map);
  crs_matrix_type SparseMatrix("matrix", numRows, values, myGraph);
  return SparseMatrix;
}

void apply_Sparse_Matrix_Kokkos(crs_matrix_type sparse_matrix,
                                Kokkos::View<vectorType **> vector_view,
                                Kokkos::View<vectorType **> result_view) {
  const data_type alpha(1.0);
  const data_type beta;
  KokkosSparse::spmv("N", alpha, sparse_matrix, vector_view, beta, result_view);
}

crs_matrix_type get_sparse_matrix(index_type sblk) {
  index_type numRows = sblk * 2;
  index_type numNNZ = sblk * 4; // 2 non-zero per row/col
  index_type *row_map_ptr =
      (index_type *)malloc((numRows + 1) * sizeof(index_type));
  index_type *entries_ptr = (index_type *)malloc(numNNZ * sizeof(index_type));
  data_type *values_ptr = (data_type *)malloc(numNNZ * sizeof(data_type));

  std::size_t count = 0;
  std::size_t i;

  for (i = 0; i < sblk; ++i) {
    row_map_ptr[i] = count;
    entries_ptr[count] = i;
    values_ptr[count] = static_cast<data_type>(1.0);
    count += 1;
    entries_ptr[count] = i + sblk;
    values_ptr[count] = static_cast<data_type>(1.0);
    count += 1;
  }
  for (i = sblk; i < 2 * sblk; ++i) {
    row_map_ptr[i] = count;
    entries_ptr[count] = i - sblk;
    values_ptr[count] = static_cast<data_type>(1.0);
    count += 1;
    entries_ptr[count] = i;
    values_ptr[count] = static_cast<data_type>(1.0);
    count += 1;
  }
  row_map_ptr[++i] = count;
  crs_matrix_type matrix = create_Kokkos_Sparse_Matrix(
      row_map_ptr, numRows, entries_ptr, values_ptr, numNNZ);
  return matrix;
}
