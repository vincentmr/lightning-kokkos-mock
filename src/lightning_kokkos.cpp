#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#include "Gates.hpp"
#include "SparseGates.hpp"

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
using Layout = Kokkos::LayoutLeft;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;

int get_itype(const std::string &type);

int main(int argc, char *argv[]) {
  std::size_t num_qubits = 20; // number of qubits
  std::size_t S = 0;           // total size 2**num_qubits
  std::size_t T = 0;           // target in [0, num_qubits-1]
  std::size_t nrepeat = 100;   // number of repeats of the test
  std::string type = "ref";

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-T") == 0)) {
      T = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "--type") == 0)) {
      type = static_cast<std::string>(argv[++i]);
    } else if ((strcmp(argv[i], "-S") == 0) ||
               (strcmp(argv[i], "-Size") == 0)) {
      num_qubits = atof(argv[++i]);
    } else if (strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  H*V Options:\n");
      printf("  -T <int>:          target qubit index (default: 0)\n");
      printf("  -Size (-S) <int>:  exponent num, determines number of "
             "qubits "
             "size 2^num (default: 2^20 = 1024*1024 )\n");
      printf("  -nrepeat <int>:    number of repetitions (default: "
             "100)\n");
      printf("  -help (-h):        print this message\n\n");
      exit(1);
    }
  }

  int itype = get_itype(type);
  S = pow(2, num_qubits);

  Kokkos::initialize(argc, argv);
  {

    // Allocate state vector and gate matrix.
    typedef Kokkos::View<vectorType *, Layout, MemSpace> ViewVectorType;
    typedef Kokkos::View<vectorType **, Layout, MemSpace> ViewMatrixType;
    ViewVectorType sv0("sv0", S);
    ViewVectorType sv1("sv1", S);
    constexpr int mats = 4;
    ViewVectorType mat("mat", mats);

    // Create host mirrors of device views.
    ViewVectorType::HostMirror mat_h = Kokkos::create_mirror_view(mat);
    ViewVectorType::HostMirror sv0_h = Kokkos::create_mirror_view(sv0);
    ViewVectorType::HostMirror sv1_h = Kokkos::create_mirror_view(sv1);

    // Initialize matrix and vector on host.
    Kokkos::deep_copy(mat_h, static_cast<vectorType>(1.0));
    Kokkos::deep_copy(sv0_h, static_cast<vectorType>(1.0));
    Kokkos::deep_copy(sv1_h, static_cast<vectorType>(1.0));

    // Deep copy host views to device views.
    Kokkos::deep_copy(mat, mat_h);
    Kokkos::deep_copy(sv0, sv0_h);
    Kokkos::deep_copy(sv1, sv1_h);

    std::vector<size_t> wires = {T};
    std::size_t nblk = std::pow(2, num_qubits - T - 1);
    std::size_t sblk = std::pow(2, T);

    ViewMatrixType sm0("sm0", sblk * 2, nblk);
    ViewMatrixType sm1("sm1", sblk * 2, nblk);
    ViewMatrixType::HostMirror sm0_h = Kokkos::create_mirror_view(sm0);
    ViewMatrixType::HostMirror sm1_h = Kokkos::create_mirror_view(sm1);
    Kokkos::deep_copy(sm0_h, static_cast<vectorType>(1.0));
    Kokkos::deep_copy(sm1_h, static_cast<vectorType>(1.0));
    Kokkos::deep_copy(sm0, sm0_h);
    Kokkos::deep_copy(sm1, sm1_h);
    // crs_matrix_type matrix;
    // if (itype == 7) { // spmv
    //   matrix = get_sparse_matrix(static_cast<index_type>(sblk));
    // }

    // Timer products.
    Kokkos::fence();
    Kokkos::Timer timer;

    for (int repeat = 0; repeat < nrepeat; repeat++) {

      // singleQubitOpFunctor f{singleQubitOpFunctor(sv0, num_qubits,
      // mat, wires)}; for (int i = 0; i < std::pow(2, num_qubits - 1);
      // i++) {
      //   f(i);
      // }

      if (itype == 0) { // ref
        Kokkos::parallel_for(range_policy(0, std::pow(2, num_qubits - 1)),
                             singleQubitOpFunctor(sv0, num_qubits, mat, wires));
        continue;
      }

      if (itype == 1) { // mdr
        Kokkos::parallel_for("H*sv0", mdrange_policy({0, 0}, {nblk, sblk}),
                             mdRangeFunctor(sv0, num_qubits, mat, wires));
        continue;
      }

      if (itype == 2) { // tv1
        Kokkos::parallel_for(
            "H*sv0", team_policy(nblk, Kokkos::AUTO, 32),
            threadVectorFunctor<32>(sv0, num_qubits, mat, wires));
        continue;
      }

      if (itype == 3) { // tv2
        Kokkos::parallel_for(
            "H*sv0",
            team_policy(std::pow(2, num_qubits - 1 - std::log2(1024)),
                        Kokkos::AUTO, 1024),
            threadVectorFunctor2<1024>(sv0, num_qubits, mat, wires));
        continue;
      }

      if (itype == 4) { // ttr
        Kokkos::parallel_for("H*sv0", team_policy(nblk, Kokkos::AUTO),
                             teamPolicyFunctor(sv0, num_qubits, mat, wires));
        continue;
      }

      if (itype == 5) { // tp0
        Kokkos::parallel_for(
            "teamPolicyFunctor0",
            team_policy(std::pow(2, num_qubits - 1), Kokkos::AUTO),
            teamPolicyFunctor0(sv0, num_qubits, mat, wires));
        continue;
      }
      if (itype == 6) { // tpv
        Kokkos::parallel_for(
            "teamPolicyFunctor0",
            team_policy(std::pow(2, num_qubits - 1 - std::log2(256)), 1, 256),
            teamPolicyFunctorV<256>(sv0, num_qubits, mat, wires));
        continue;
      }
      if (itype == 7) { // spmv
        // apply_Sparse_Matrix_Kokkos(matrix, sm0, sm1);
        Kokkos::parallel_for(
            range_policy(0, nblk),
            sparseSingleQubitOpFunctor(static_cast<size_t>(T), sm0, sm1));
        continue;
      }
    }
    Kokkos::fence();

    // Calculate time.
    double time = timer.seconds();

    // Calculate bandwidth.
    double Gbytes = 1.0e-9 * double(sizeof(vectorType) * S);

    // Print results (problem size, time and bandwidth in GB/s).
    printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
           time, Gbytes * nrepeat / time);
  }
  Kokkos::finalize();

  return 0;
}

int get_itype(const std::string &type) {

  if (type == "ref") {
    return 0;
  }
  if (type == "mdr") {
    return 1;
  }
  if (type == "tv1") {
    return 2;
  }
  if (type == "tv2") {
    return 3;
  }
  if (type == "ttr") {
    return 4;
  }
  if (type == "tp0") {
    return 5;
  }
  if (type == "tpv") {
    return 6;
  }
  if (type == "spmv") {
    return 7;
  }
  return -1;
}
