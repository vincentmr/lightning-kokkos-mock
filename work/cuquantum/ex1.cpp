#include <cuComplex.h>        // cuDoubleComplex
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <cstring>
#include <sys/time.h>  // gettimeofday()
#include <sys/types.h> // struct timeval

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}

void tellTime(int size, int rank, int param0, int param1, double elapsedTime) {
  if (rank == 0) {
    printf("%6d, ", size);
    printf("%10d, ", param0);
    printf("%10d, ", param1);
    printf("%12.8f\n", elapsedTime);
  }
}

int main(int argc, char *argv[]) {

  std::size_t num_qubits = 20; // number of qubits
  std::size_t S = 0;           // total size 2**num_qubits
  std::size_t T = 0;           // target in [0, num_qubits-1]
  std::size_t nrepeat = 100;   // number of repeats of the test

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-T") == 0)) {
      T = atoi(argv[++i]);
      // printf("  User T is %d\n", T);
    } else if ((strcmp(argv[i], "-S") == 0) ||
               (strcmp(argv[i], "-Size") == 0)) {
      num_qubits = atof(argv[++i]);
      // printf("  User S is %d\n", S);
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

  S = pow(2, num_qubits);

  std::size_t sblk = std::pow(2, T);
  std::size_t nblk = std::pow(2, num_qubits - T - 1);

  const int nIndexBits = num_qubits;
  const int nSvSize = S;
  const int nTargets = 1;
  const int nControls = 0;
  const int adjoint = 0;

  int targets[] = {static_cast<int>(T)};
  int controls[] = {0, 1};

  cuDoubleComplex *h_sv =
      (cuDoubleComplex *)malloc(S * sizeof(cuDoubleComplex));
  cuDoubleComplex *h_sv_result =
      (cuDoubleComplex *)malloc(S * sizeof(cuDoubleComplex));

  cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cuDoubleComplex *d_sv;
  cudaMalloc((void **)&d_sv, S * sizeof(cuDoubleComplex));

  cudaMemcpy(d_sv, h_sv, S * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  //--------------------------------------------------------------------------

  // custatevec handle initialization
  custatevecHandle_t handle;

  custatevecCreate(&handle);

  void *extraWorkspace = nullptr;
  size_t extraWorkspaceSizeInBytes = 0;

  // check the size of external workspace
  custatevecApplyMatrixGetWorkspaceSize(
      handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
      CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTargets, nControls,
      CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes);

  // allocate external workspace if necessary
  if (extraWorkspaceSizeInBytes > 0)
    cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

  // apply gate
  cudaDeviceSynchronize();
  double startTime = getTime();
  // execute SpMM
  for (int c = 0; c < nrepeat; ++c) {
    custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, nIndexBits, matrix,
                          CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint,
                          targets, nTargets, controls, nullptr, nControls,
                          CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                          extraWorkspaceSizeInBytes);
  }
  cudaDeviceSynchronize();

  double Gbytes = 1.0e-9 * double(sizeof(cuDoubleComplex) * S);
  auto time = (getTime() - startTime);
  printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
         time, Gbytes * nrepeat / time);

  // destroy handle
  custatevecDestroy(handle);

  //--------------------------------------------------------------------------

  //   cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
  //              cudaMemcpyDeviceToHost);

  //   bool correct = true;
  //   for (int i = 0; i < nSvSize; i++) {
  //     if ((h_sv[i].x != h_sv_result[i].x) || (h_sv[i].y != h_sv_result[i].y))
  //     {
  //       correct = false;
  //       break;
  //     }
  //   }

  //   if (correct)
  //     printf("example PASSED\n");
  //   else
  //     printf("example FAILED: wrong result\n");

  cudaFree(d_sv);
  if (extraWorkspaceSizeInBytes)
    cudaFree(extraWorkspace);

  return EXIT_SUCCESS;
}