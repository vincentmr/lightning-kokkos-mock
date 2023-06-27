/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

// #include "Util.hpp"

// #include <simd.hpp>
// #ifdef KOKKOS_ENABLE_CUDA
// using simd_t = simd::simd<double, simd::simd_abi::cuda_warp<32>>;
// #else
// using simd_t = simd::simd<double, simd::simd_abi::native>;
// #endif
// using simd_storage_t = simd_t::storage_type;

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
using range_policy = Kokkos::RangePolicy<ExecSpace>;
using mdrange_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using layout = Kokkos::LayoutLeft;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;
// using vectorType = Kokkos::complex<double>;
#include <complex>
using vectorType = std::complex<double>;
// using vectorType = double;

void checkSizes(std::size_t &num_qubits, std::size_t &S, std::size_t &T,
                std::size_t &nrepeat);

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

//  struct singleQubitSimdFunctor {

//   Kokkos::View<vectorType *> arr;
//   Kokkos::View<vectorType *> matrix;

//   std::size_t rev_wire;
//   std::size_t rev_wire_shift;
//   std::size_t wire_parity;
//   std::size_t wire_parity_inv;

//   singleQubitSimdFunctor(
//       Kokkos::View<vectorType *> arr_, std::size_t
//       num_qubits, const Kokkos::View<vectorType *> matrix_,
//       const std::vector<size_t> &wires) {
//     arr = arr_;
//     matrix = matrix_;
//     rev_wire = num_qubits - wires[0] - 1;
//     rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
//     wire_parity = fillTrailingOnes(rev_wire);
//     wire_parity_inv = fillLeadingOnes(rev_wire + 1);
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t k) const {
//     const std::size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity &
//     k); const std::size_t i1 = i0 | rev_wire_shift; const
//     vectorType v0 = arr[i0]; const
//     vectorType v1 = arr[i1]; arr[i0] = matrix[0B00] * v0 +
//     matrix[0B01] * v1; arr[i1] = matrix[0B10] * v0 + matrix[0B11] * v1;
//   }
// };

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
        Kokkos::TeamThreadRange(teamMember, 0, sblk / V), [&](const int j) {
          const std::size_t jV = j * V;
          Kokkos::parallel_for(
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

int main(int argc, char *argv[]) {
  std::size_t num_qubits = -1; // number of qubits
  std::size_t S = -1;          // total size 2^22
  std::size_t T = -1;          // target
  std::size_t nrepeat = 100;   // number of repeats of the test

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-T") == 0)) {
      T = atoi(argv[++i]);
      // printf("  User T is %d\n", T);
    } else if ((strcmp(argv[i], "-S") == 0) ||
               (strcmp(argv[i], "-Size") == 0)) {
      num_qubits = atof(argv[++i]);
      S = pow(2, num_qubits);
      // printf("  User S is %d\n", S);
    } else if (strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  H*V Options:\n");
      printf("  -T <int>:          target qubit index (default: 0)\n");
      printf("  -Size (-S) <int>:  exponent num, determines number of qubits "
             "size 2^num (default: 2^20 = 1024*1024 )\n");
      printf("  -nrepeat <int>:    number of repetitions (default: 100)\n");
      printf("  -help (-h):        print this message\n\n");
      exit(1);
    }
  }

  // Check sizes.
  checkSizes(num_qubits, S, T, nrepeat);

  Kokkos::initialize(argc, argv);
  {

    // Allocate y, x vectors and Matrix A on device.
    typedef Kokkos::View<vectorType *, layout, MemSpace> ViewVectorType;
    ViewVectorType sv0("sv0", S);
    ViewVectorType sv1("sv1", S);
    constexpr int mats = 4;
    ViewVectorType mat("mat", mats);

    // Create host mirrors of device views.
    ViewVectorType::HostMirror mat_h = Kokkos::create_mirror_view(mat);
    ViewVectorType::HostMirror sv0_h = Kokkos::create_mirror_view(sv0);

    // Initialize y vector on host.
    for (int i = 0; i < mats; ++i) {
      mat_h[i] = static_cast<vectorType>(1);
    }
    for (int i = 0; i < S; ++i) {
      sv0_h[i] = static_cast<vectorType>(1);
    }

    // Deep copy host views to device views.
    Kokkos::deep_copy(mat, mat_h);
    Kokkos::deep_copy(sv0, sv0_h);
    // std::unique_ptr<ViewVectorType> data_ =
    //     std::make_unique<ViewVectorType>("data_", std::pow(2, num_qubits));
    // data_ = sv0;

    std::vector<size_t> wires = {T};

    // Timer products.
    Kokkos::Timer timer;

    for (int repeat = 0; repeat < nrepeat; repeat++) {
      // Application: <y,Ax> = y^T*A*x

      // singleQubitOpFunctor f{singleQubitOpFunctor(sv0, num_qubits, mat,
      // wires)}; for (int i = 0; i < std::pow(2, num_qubits - 1); i++) {
      //   f(i);
      // }

      Kokkos::parallel_for(range_policy(0, std::pow(2, num_qubits - 1)),
                           singleQubitOpFunctor(sv0, num_qubits, mat, wires));

      // std::size_t nblk = std::pow(2, num_qubits - wires[0] - 1);
      // std::size_t sblk = std::pow(2, wires[0]);
      // Kokkos::parallel_for("H*sv0", mdrange_policy({0, 0}, {nblk, sblk}),
      //                      mdRangeFunctor(sv0, num_qubits, mat,
      //                      wires));

      // std::size_t nblk = std::pow(2, num_qubits - wires[0] - 1);
      // std::size_t sblk = std::pow(2, wires[0]);
      // Kokkos::parallel_for(
      //     "H*sv0", mdrange_policy({0, 0}, {nblk, sblk}),
      //     KOKKOS_LAMBDA(const std::size_t i, const std::size_t j) {
      //       const std::size_t offset = i * sblk * 2;
      //       const std::size_t i0 = j + offset;
      //       const std::size_t i1 = i0 + sblk;
      //       const vectorType v0 = sv0[i0];
      //       const vectorType v1 = sv0[i1];
      //       sv0[i0] = mat[0B00] * v0 + mat[0B01] * v1;
      //       sv0[i1] = mat[0B10] * v0 + mat[0B11] * v1;
      //     });

      // std::size_t nblk = std::pow(2, num_qubits - wires[0] - 1);
      // Kokkos::parallel_for(
      //     "H*sv0", team_policy(nblk, Kokkos::AUTO, 32),
      //     threadVectorFunctor<32>(sv0, num_qubits, mat, wires));

      // Kokkos::parallel_for(
      //     "H*sv0",
      //     team_policy(std::pow(2, num_qubits - 1 - 5), Kokkos::AUTO, 32),
      //     threadVectorFunctor2<32>(sv0, num_qubits, mat, wires));

      // std::size_t nblk = std::pow(2, num_qubits - wires[0] - 1);
      // Kokkos::parallel_for(
      //     "H*sv0", team_policy(nblk, Kokkos::AUTO),
      //     teamPolicyFunctor(sv0, num_qubits, mat, wires));

      // std::size_t nblk = std::pow(2, num_qubits - wires[0] - 1);
      // std::size_t sblk = std::pow(2, wires[0]);
      // Kokkos::parallel_for(
      //     "H*sv0", team_policy(nblk, Kokkos::AUTO),
      //     KOKKOS_LAMBDA(const member_type &teamMember) {
      //       const int i = teamMember.league_rank();
      //       const std::size_t offset = i * sblk * 2;
      //       Kokkos::parallel_for(
      //           Kokkos::TeamThreadRange(teamMember, 0, sblk),
      //           // Kokkos::TeamVectorRange(teamMember, 0, sblk),
      //           [=](const int j) {
      //             const std::size_t i0 = j + offset;
      //             const std::size_t i1 = i0 + sblk;
      //             const vectorType v0 = sv0[i0];
      //             const vectorType v1 = sv0[i1];
      //             sv0[i0] = mat[0B00] * v0 + mat[0B01] * v1;
      //             sv0[i1] = mat[0B10] * v0 + mat[0B11] * v1;
      //           });
      //     });

      // Output result.
      // if (repeat == (nrepeat - 1)) {
      //   printf("  Computed result for %d x %d is %lf\n", N, M, result);
      // }

      // const double solution = (double)N * (double)M;

      // if (result != solution) {
      //   printf("  Error: result( %lf ) != solution( %lf )\n", result,
      //   solution);
      // }
    }

    Kokkos::fence();
    // Calculate time.
    double time = timer.seconds();

    // Calculate bandwidth.
    // Each matrix A row (each of length M) is read once.
    // The x vector (of length M) is read N times.
    // The y vector (of length N) is read once.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
    double Gbytes = 1.0e-9 * double(sizeof(vectorType) * S);

    // Print results (problem size, time and bandwidth in GB/s).
    // printf("  S( %12d ) nrepeat ( %12d ) problem( %12g MB ) time( %12g s )
    // "
    //        "bandwidth( %12g GB/s )\n",
    //        S, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
    printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
           time, Gbytes * nrepeat / time);
  }
  Kokkos::finalize();

  return 0;
}

void checkSizes(std::size_t &num_qubits, std::size_t &S, std::size_t &T,
                std::size_t &nrepeat) {

  if (num_qubits == -1) {
    num_qubits = 20;
    S = pow(2, num_qubits);
  }

  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of
  // N and M.
  if (S == -1) {
    S = pow(2, num_qubits);
  }

  // If both N and M are undefined, fix row length to the smaller of S and
  // 2^10 = 1024.
  if (T == -1) {
    T = 0;
  }

  // printf("  Total size S = %d T = %d \n", S, T);

  // Check sizes.
  if ((S <= 0) || (T < 0) || (nrepeat <= 0)) {
    printf("  Sizes must be greater than 0.\n");
    exit(1);
  }
}
