#pragma once

#include <cassert>
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

using index_type = long int;

void set_sparse_data(const index_type sblk,
                     Kokkos::View<index_type *>::HostMirror &row_map_h,
                     Kokkos::View<index_type *>::HostMirror &entries_h,
                     Kokkos::View<vectorType *>::HostMirror &values_h) {
  std::size_t count = 0;
  std::size_t i;
  const vectorType SC_ONE = Kokkos::ArithTraits<vectorType>::one();

  for (i = 0; i < sblk; i++) {
    row_map_h(i) = count;
    entries_h(count) = i;
    values_h(count) = SC_ONE;
    count += 1;
    entries_h(count) = i + sblk;
    values_h(count) = SC_ONE;
    count += 1;
  }
  for (i = sblk; i < 2 * sblk; i++) {
    row_map_h(i) = count;
    entries_h(count) = i - sblk;
    values_h(count) = SC_ONE;
    count += 1;
    entries_h(count) = i;
    values_h(count) = SC_ONE;
    count += 1;
  }
}

void apply_Sparse_Matrix_Kokkos(Kokkos::View<index_type *> row_map,
                                Kokkos::View<index_type *> entries,
                                Kokkos::View<vectorType *> values,
                                Kokkos::View<vectorType *> x,
                                Kokkos::View<vectorType *> y) {
  const vectorType SC_ZERO = Kokkos::ArithTraits<vectorType>::zero();
  size_t vector_size = x.extent(0);
  size_t count = 0;
  // assert(vector_size == row_map.extent(0)-2);
  for (index_type i = 0; i < vector_size; i++) {
    y(i) = SC_ZERO;
    for (index_type j = 0; j < row_map(i + 1) - row_map(i); j++) {
      y(i) += values(count) * x(entries(count));
      count++;
    }
  }
}

struct sparseSingleQubitOpFunctor {

  Kokkos::View<index_type *> row_map;
  Kokkos::View<index_type *> entries;
  Kokkos::View<vectorType *> values;
  Kokkos::View<vectorType **> x;
  Kokkos::View<vectorType **> y;

  sparseSingleQubitOpFunctor(const size_t target,
                             Kokkos::View<vectorType **> x_,
                             Kokkos::View<vectorType **> y_) {
    x = x_;
    y = y_;

    size_t sblk = std::pow(2, target);
    Kokkos::resize(row_map, 2 * sblk + 1);
    Kokkos::resize(entries, 4 * sblk);
    Kokkos::resize(values, 4 * sblk);
    Kokkos::View<index_type *>::HostMirror row_map_h =
        Kokkos::create_mirror_view(row_map);
    Kokkos::View<index_type *>::HostMirror entries_h =
        Kokkos::create_mirror_view(entries);
    Kokkos::View<vectorType *>::HostMirror values_h =
        Kokkos::create_mirror_view(values);
    set_sparse_data(target / 2, row_map_h, entries_h, values_h);
    Kokkos::deep_copy(row_map, row_map_h);
    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t j) const {
    auto xj = subview(x, Kokkos::ALL, j);
    auto yj = subview(y, Kokkos::ALL, j);
    apply_Sparse_Matrix_Kokkos(row_map, entries, values, xj, yj);
  }
};
