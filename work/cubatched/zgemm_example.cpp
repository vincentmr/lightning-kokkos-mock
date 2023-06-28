/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
// #include <cublas.h>           // cublasZgemm
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <cassert>
#include <cmath>
#include <sys/time.h>  // gettimeofday()
#include <sys/types.h> // struct timeval

#include <complex>
#include <cuComplex.h>
#include <vector>

using cuZ = cuDoubleComplex;

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      printf("CUBLAS API failed at line %d with error: (%d)\n", __LINE__,      \
             status);                                                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

void get_A_matrix(const int sblk, int *hA_csrOffsets, int *hA_columns,
                  cuDoubleComplex *hA_values);
double getTime();
void tellTime(int size, int rank, int param0, int param1, double elapsedTime);

int main(int argc, char *argv[]) {

  cublasOperation_t trans = CUBLAS_OP_N; // fixed

  //--------------------------------------------------------------------------
  // READ COMMAND LINE ARGUMENTS
  std::size_t log_S = 20;    // log size
  std::size_t S = 0;         // total size 2**log_S
  std::size_t T = 0;         // target (block size)
  std::size_t nrepeat = 100; // number of repeats of the test
  std::size_t batch_count = 1024;

  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-T") == 0)) {
      T = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-B") == 0)) {
      batch_count = atof(argv[++i]);
    } else if ((strcmp(argv[i], "-S") == 0)) {
      log_S = atof(argv[++i]);
    } else if (strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("  H*V Options:\n");
      printf("  -T <int>:          target (default: 0)\n");
      printf("  -Size (-S) <int>:  log size (num)"
             "size 2**num (default: 2**20 = 1024*1024 )\n");
      printf("  -nrepeat <int>:    number of repetitions (default: "
             "100)\n");
      printf("  -help (-h):        print this message\n\n");
      exit(1);
    }
  }
  S = pow(2, log_S);

  std::size_t sblk = std::pow(2, T);             // block size div-by-2
  std::size_t nblk = std::pow(2, log_S - T - 1); // number of blocks
  if (batch_count > nblk)
    batch_count = nblk;

  //--------------------------------------------------------------------------
  // HOST PROBLEM DEFINITION

  int A_num_rows = sblk * 2;
  int A_num_cols = sblk * 2;
  int B_num_rows = A_num_cols;
  int B_num_cols = nblk / batch_count;
  int A_size = A_num_rows * A_num_cols;
  int B_size = B_num_rows * B_num_cols;
  int C_size = A_num_rows * B_num_cols;
  int ldb = B_num_rows;
  int ldc = A_num_rows;

  std::vector<std::vector<cuZ>> hA(batch_count, std::vector<cuZ>(A_size));
  std::vector<std::vector<cuZ>> hB(batch_count, std::vector<cuZ>(B_size));
  std::vector<std::vector<cuZ>> hC(batch_count, std::vector<cuZ>(C_size));
  cuZ alpha = {1.0, 0.0};
  cuZ beta = {0.0, 0.0};

  //--------------------------------------------------------------------------
  // DEVICE MEMORY MANAGEMENT
  CHECK_CUDA(cudaSetDevice(0));

  std::vector<cuZ *> d_A(batch_count, nullptr);
  std::vector<cuZ *> d_B(batch_count, nullptr);
  std::vector<cuZ *> d_C(batch_count, nullptr);
  cuZ **d_A_array = nullptr;
  cuZ **d_B_array = nullptr;
  cuZ **d_C_array = nullptr;

  for (int i = 0; i < batch_count; i++) {
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A[i]),
                          sizeof(cuZ) * hA[i].size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_B[i]),
                          sizeof(cuZ) * hB[i].size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_C[i]),
                          sizeof(cuZ) * hC[i].size()));
  }
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_A_array),
                        sizeof(cuZ *) * batch_count));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_B_array),
                        sizeof(cuZ *) * batch_count));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_C_array),
                        sizeof(cuZ *) * batch_count));

  for (int i = 0; i < batch_count; i++) {
    CHECK_CUDA(cudaMemcpy(d_A[i], hA[i].data(), sizeof(cuZ) * hA[i].size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B[i], hB[i].data(), sizeof(cuZ) * hB[i].size(),
                          cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaMemcpy(d_A_array, d_A.data(), sizeof(cuZ *) * batch_count,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_array, d_B.data(), sizeof(cuZ *) * batch_count,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C_array, d_C.data(), sizeof(cuZ *) * batch_count,
                        cudaMemcpyHostToDevice));

  //--------------------------------------------------------------------------
  // CUBLAS APIs
  cublasHandle_t handle = NULL;
  CHECK_CUBLAS(cublasCreate(&handle))

  // execute SpMM
  cudaDeviceSynchronize();
  double startTime = getTime();
  for (int c = 0; c < nrepeat; ++c) {
    CHECK_CUBLAS(cublasZgemmBatched(handle, trans, trans, A_num_rows,
                                    B_num_cols, A_num_cols, &alpha, d_A_array,
                                    A_num_rows, d_B_array, B_num_rows, &beta,
                                    d_C_array, A_num_rows, batch_count))
  }
  cudaDeviceSynchronize();

  double Gbytes = 1.0e-9 * double(sizeof(cuZ) * S);
  auto time = (getTime() - startTime);
  printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
         time, Gbytes * nrepeat / time);

  //--------------------------------------------------------------------------
  // CLEANUP
  CHECK_CUBLAS(cublasDestroy(handle))

  // device memory deallocation
  CHECK_CUDA(cudaFree(d_A_array));
  CHECK_CUDA(cudaFree(d_B_array));
  CHECK_CUDA(cudaFree(d_C_array));
  for (int i = 0; i < batch_count; i++) {
    CHECK_CUDA(cudaFree(d_A[i]));
    CHECK_CUDA(cudaFree(d_B[i]));
    CHECK_CUDA(cudaFree(d_C[i]));
  }

  return EXIT_SUCCESS;
}

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
