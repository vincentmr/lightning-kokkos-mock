#pragma once

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

// #include "Util.hpp"

// using vectorType = double;
using vectorType = Kokkos::complex<double>;
// using vectorType = std::complex<double>;

/**
 * @brief Fill ones from LSB to rev_wire
 */
inline size_t fillTrailingOnes(size_t pos) {
  return (pos == 0) ? 0 : (~size_t(0) >> (CHAR_BIT * sizeof(size_t) - pos));
}

/**
 * @brief Fill ones from MSB to pos
 */
inline auto constexpr fillLeadingOnes(size_t pos) -> size_t {
  return (~size_t(0)) << pos;
}

struct singleQubitOpFunctor {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t rev_wire;
  std::size_t rev_wire_shift;
  std::size_t wire_parity;
  std::size_t wire_parity_inv;

  singleQubitOpFunctor(Kokkos::View<vectorType *> arr_, std::size_t num_qubits,
                       const Kokkos::View<vectorType *> matrix_,
                       const std::vector<size_t> &wires) {
    arr = arr_;
    matrix = matrix_;
    rev_wire = num_qubits - wires[0] - 1;
    rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    wire_parity = fillTrailingOnes(rev_wire);
    wire_parity_inv = fillLeadingOnes(rev_wire + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t k) const {
    const std::size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
    const std::size_t i1 = i0 | rev_wire_shift;
    const vectorType v0 = arr[i0];
    const vectorType v1 = arr[i1];
    arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
    arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
  }
};

struct teamPolicyFunctor {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t nblk;
  std::size_t sblk;

  teamPolicyFunctor(Kokkos::View<vectorType *> arr_, std::size_t num_qubits_,
                    const Kokkos::View<vectorType *> matrix_,
                    const std::vector<size_t> &wires_) {
    arr = arr_;
    matrix = matrix_;
    nblk = std::pow(2, num_qubits_ - wires_[0] - 1);
    sblk = std::pow(2, wires_[0]);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type &teamMember) const {
    const int i = teamMember.league_rank();
    const std::size_t offset = i * sblk * 2;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(teamMember, 0, sblk),
        // Kokkos::TeamVectorRange(teamMember, 0, sblk),
        KOKKOS_LAMBDA(const int j) {
          const std::size_t i0 = j + offset;
          const std::size_t i1 = i0 + sblk;
          const vectorType v0 = arr[i0];
          const vectorType v1 = arr[i1];
          arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
          arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
        });
  }
};

struct teamPolicyFunctor0 {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t rev_wire;
  std::size_t rev_wire_shift;
  std::size_t wire_parity;
  std::size_t wire_parity_inv;

  teamPolicyFunctor0(Kokkos::View<vectorType *> arr_, std::size_t num_qubits,
                     const Kokkos::View<vectorType *> matrix_,
                     const std::vector<size_t> &wires) {
    arr = arr_;
    matrix = matrix_;
    rev_wire = num_qubits - wires[0] - 1;
    rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    wire_parity = fillTrailingOnes(rev_wire);
    wire_parity_inv = fillLeadingOnes(rev_wire + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type &teamMember) const {
    const int k = teamMember.league_rank();
    const std::size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
    const std::size_t i1 = i0 | rev_wire_shift;
    const vectorType v0 = arr[i0];
    const vectorType v1 = arr[i1];
    arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
    arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
  }
};

template <std::size_t V> struct teamPolicyFunctorV {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t rev_wire;
  std::size_t rev_wire_shift;
  std::size_t wire_parity;
  std::size_t wire_parity_inv;

  teamPolicyFunctorV(Kokkos::View<vectorType *> arr_, std::size_t num_qubits,
                     const Kokkos::View<vectorType *> matrix_,
                     const std::vector<size_t> &wires) {
    arr = arr_;
    matrix = matrix_;
    rev_wire = num_qubits - wires[0] - 1;
    rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    wire_parity = fillTrailingOnes(rev_wire);
    wire_parity_inv = fillLeadingOnes(rev_wire + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type &teamMember) const {
    const int k = teamMember.league_rank() * V;
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(teamMember, V), KOKKOS_LAMBDA(const std::size_t kk) {
          const std::size_t i0 =
              (((k + kk) << 1U) & wire_parity_inv) | (wire_parity & (k + kk));
          const std::size_t i1 = i0 | rev_wire_shift;
          const vectorType v0 = arr[i0];
          const vectorType v1 = arr[i1];
          arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
          arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
        });
  }
};

template <std::size_t V> struct threadVectorFunctor {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t nblk;
  std::size_t sblk;

  threadVectorFunctor(Kokkos::View<vectorType *> arr_, std::size_t num_qubits_,
                      const Kokkos::View<vectorType *> matrix_,
                      const std::vector<size_t> &wires_) {
    arr = arr_;
    matrix = matrix_;
    nblk = std::pow(2, num_qubits_ - wires_[0] - 1);
    sblk = std::pow(2, wires_[0]);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type &teamMember) const {
    const int i = teamMember.league_rank();
    const std::size_t offset = i * sblk * 2;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(teamMember, 0, (sblk / V > 0) ? sblk / V : 1),
        [&](const int j) {
          const std::size_t jV = j * V;
          Kokkos::parallel_for( // Kokkos::TeamVectorRange(teamMember, V),
              Kokkos::ThreadVectorRange(teamMember, V), [&](const int ii) {
                const std::size_t i0 = jV + ii + offset;
                const std::size_t i1 = i0 + sblk;
                const vectorType v0 = arr[i0];
                const vectorType v1 = arr[i1];
                arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
                arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
              });
        });
  }
};

template <std::size_t V> struct threadVectorFunctor2 {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t rev_wire;
  std::size_t rev_wire_shift;
  std::size_t wire_parity;
  std::size_t wire_parity_inv;

  threadVectorFunctor2(Kokkos::View<vectorType *> arr_, std::size_t num_qubits,
                       const Kokkos::View<vectorType *> matrix_,
                       const std::vector<size_t> &wires) {
    arr = arr_;
    matrix = matrix_;
    rev_wire = num_qubits - wires[0] - 1;
    rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    wire_parity = fillTrailingOnes(rev_wire);
    wire_parity_inv = fillLeadingOnes(rev_wire + 1);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type &teamMember) const {
    const int k = teamMember.league_rank() * V;
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(teamMember, V), KOKKOS_LAMBDA(int kk) {
          const int i0 =
              (((k + kk) << 1U) & wire_parity_inv) | (wire_parity & (k + kk));
          const int i1 = i0 | rev_wire_shift;
          const vectorType v0 = arr[i0];
          const vectorType v1 = arr[i1];
          arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
          arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
        });
  }
};

struct mdRangeFunctor {

  Kokkos::View<vectorType *> arr;
  Kokkos::View<vectorType *> matrix;

  std::size_t nblk;
  std::size_t sblk;

  mdRangeFunctor(Kokkos::View<vectorType *> arr_, std::size_t num_qubits_,
                 const Kokkos::View<vectorType *> matrix_,
                 const std::vector<size_t> &wires_) {
    arr = arr_;
    matrix = matrix_;
    nblk = std::pow(2, num_qubits_ - wires_[0] - 1);
    sblk = std::pow(2, wires_[0]);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i, const std::size_t j) const {
    const std::size_t offset = i * sblk * 2;
    const std::size_t i0 = j + offset;
    const std::size_t i1 = i0 + sblk;
    const vectorType v0 = arr[i0];
    const vectorType v1 = arr[i1];
    arr[i0] = matrix[0B00] * v0 + matrix[0B01] * v1;
    arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
  };
};
