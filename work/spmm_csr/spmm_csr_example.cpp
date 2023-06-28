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

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <cassert>
#include <cmath>
#include <sys/time.h>  // gettimeofday()
#include <sys/types.h> // struct timeval

#include <complex>
#include <cuComplex.h>

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

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

void get_A_matrix(const int sblk, int *hA_csrOffsets, int *hA_columns,
                  cuDoubleComplex *hA_values);
double getTime();
void tellTime(int size, int rank, int param0, int param1, double elapsedTime);

int main(int argc, char *argv[]) {

  //--------------------------------------------------------------------------
  // READ COMMAND LINE ARGUMENTS
  std::size_t log_S = 20;    // log size
  std::size_t S = 0;         // total size 2**log_S
  std::size_t T = 0;         // target (block size)
  std::size_t nrepeat = 100; // number of repeats of the test

  for (int i = 0; i < argc; i++) {
    if ((strcmp(argv[i], "-T") == 0)) {
      T = atoi(argv[++i]);
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

  //--------------------------------------------------------------------------
  // HOST PROBLEM DEFINITION
  int A_num_rows = sblk * 2;
  int A_num_cols = sblk * 2;
  int A_nnz = sblk * 4; // 2 non-zero per row/col
  int B_num_rows = A_num_cols;
  int B_num_cols = nblk;
  int ldb = B_num_rows;
  int ldc = A_num_rows;
  int B_size = ldb * B_num_cols;
  int C_size = ldc * B_num_cols;
  int *hA_csrOffsets;
  int *hA_columns;
  cuZ *hA_values;
  cuZ *hB;
  cuZ *hC;
  cuZ *hC_result;
  cuZ alpha = {1.0, 0.0};
  cuZ beta = {0.0, 0.0};

  hA_csrOffsets = (int *)malloc((A_num_rows + 1) * sizeof(int));
  hA_columns = (int *)malloc(A_nnz * sizeof(int));
  hA_values = (cuZ *)malloc(A_nnz * sizeof(cuZ));
  hB = (cuZ *)malloc(std::pow(2, log_S) * sizeof(cuZ));
  hC = (cuZ *)malloc(std::pow(2, log_S) * sizeof(cuZ));
  hC_result = (cuZ *)malloc(std::pow(2, log_S) * sizeof(cuZ));

  get_A_matrix(sblk, hA_csrOffsets, hA_columns, hA_values);

  //--------------------------------------------------------------------------
  // DEVICE MEMORY MANAGEMENT
  int *dA_csrOffsets, *dA_columns;
  cuZ *dA_values, *dB, *dC;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(
      cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(cuZ)))
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(cuZ)))
  CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(cuZ)))

  CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(cuZ),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(cuZ), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(cuZ), cudaMemcpyHostToDevice))

  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                   dA_csrOffsets, dA_columns, dA_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F))
  // Create dense matrix B
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                     CUDA_C_64F, CUSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                     CUDA_C_64F, CUSPARSE_ORDER_COL))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG1, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute SpMM
  cudaDeviceSynchronize();
  double startTime = getTime();
  for (int c = 0; c < nrepeat; ++c) {
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, CUDA_C_64F,
                                CUSPARSE_SPMM_CSR_ALG1, dBuffer))
  }
  cudaDeviceSynchronize();

  double Gbytes = 1.0e-9 * double(sizeof(cuZ) * S);
  auto time = (getTime() - startTime);
  printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
         time, Gbytes * nrepeat / time);

  //--------------------------------------------------------------------------
  // CLEANUP
  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))

  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_csrOffsets))
  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))

  // host memory deallocation
  free(hA_csrOffsets);
  free(hA_columns);
  free(hA_values);
  free(hB);
  free(hC);
  free(hC_result);

  return EXIT_SUCCESS;
}

void get_A_matrix(const int sblk, int *hA_csrOffsets, int *hA_columns,
                  cuZ *hA_values) {
  std::size_t count = 0;
  std::size_t i;
  for (i = 0; i < sblk; ++i) {
    hA_csrOffsets[i] = count;
    hA_columns[count] = i;
    hA_values[count] = {1.0, 0.0};
    count += 1;
    hA_columns[count] = i + sblk;
    hA_values[count] = {1.0, 0.0};
    count += 1;
  }
  for (i = sblk; i < 2 * sblk; ++i) {
    hA_csrOffsets[i] = count;
    hA_columns[count] = i - sblk;
    hA_values[count] = {1.0, 0.0};
    count += 1;
    hA_columns[count] = i;
    hA_values[count] = {1.0, 0.0};
    count += 1;
  }
  hA_csrOffsets[++i] = count;
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
