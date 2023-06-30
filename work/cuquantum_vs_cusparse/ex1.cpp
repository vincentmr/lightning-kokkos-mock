#include <cuComplex.h>        // cuZ
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <cstring>
#include <sys/time.h>  // gettimeofday()
#include <sys/types.h> // struct timeval

#include <cusparse.h> // cusparseSpMM

using cuZ = cuDoubleComplex;
using cuI = std::uint64_t;

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

void get_A_matrix(const int sblk, cuZ matrix[], cuI *hA_csrOffsets,
                  cuI *hA_columns, cuDoubleComplex *hA_values);
double getTime();
void apply_cuquantum(size_t num_qubits, size_t S, size_t T, cuZ *h_sv,
                     cuZ *h_sv_result, cuZ matrix[], size_t nrepeat);
int apply_cusparse(size_t num_qubits, size_t S, size_t T, cuZ *h_sv,
                   cuZ *h_sv_result, cuZ matrix[], size_t nrepeat);

int main(int argc, char *argv[]) {

  std::size_t num_qubits = 20; // number of qubits
  std::size_t S = 0;           // total size 2**num_qubits
  std::size_t T = 0;           // target in [0, num_qubits-1]
  std::size_t nrepeat = 10;    // number of repeats of the test

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

  cuZ *h_sv = (cuZ *)malloc(S * sizeof(cuZ));
  for (int i = 0; i < S; i++) {
    h_sv[i] = {0.0, static_cast<double>(i)};
  }
  cuZ *h_sv_result = (cuZ *)malloc(S * sizeof(cuZ));
  for (int i = 0; i < S; i += 2) {
    h_sv_result[i] = {0.0, static_cast<double>(i + 1)};
  }
  for (int i = 1; i < S; i += 2) {
    h_sv_result[i] = {0.0, static_cast<double>(i - 1)};
  }

  cuZ matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  apply_cuquantum(num_qubits, S, T, h_sv, h_sv_result, matrix, nrepeat);
  apply_cusparse(num_qubits, S, T, h_sv, h_sv_result, matrix, nrepeat);
  free(h_sv);
  free(h_sv_result);
  return EXIT_SUCCESS;
}

int apply_cusparse(size_t num_qubits, size_t S, size_t T, cuZ *h_sv,
                   cuZ *h_sv_result, cuZ matrix[], size_t nrepeat) {
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  cusparseSpMMAlg_t algo = (order == CUSPARSE_ORDER_ROW)
                               ? CUSPARSE_SPMM_CSR_ALG2
                               : CUSPARSE_SPMM_CSR_ALG1;
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE; // fixed

  std::size_t sblk = std::pow(2, T);                  // block size div-by-2
  std::size_t nblk = std::pow(2, num_qubits - T - 1); // number of blocks

  cuI A_num_rows = sblk * 2;
  cuI A_num_cols = sblk * 2;
  cuI A_nnz = sblk * 4; // 2 non-zero per row/col
  cuI B_num_rows = A_num_cols;
  cuI B_num_cols = nblk;
  cuI B_size = B_num_rows * B_num_cols;
  cuI C_size = A_num_rows * B_num_cols;
  cuI ldb = (order == CUSPARSE_ORDER_ROW) ? B_size / B_num_rows : B_num_rows;
  cuI ldc = (order == CUSPARSE_ORDER_ROW) ? C_size / A_num_rows : A_num_rows;
  cuI *hA_csrOffsets;
  cuI *hA_columns;
  cuZ *hA_values;
  cuZ *hB;
  cuZ *hC;
  cuZ *hC_result;
  cuZ alpha = {1.0, 0.0};
  cuZ beta = {0.0, 0.0};

  // cuZ matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
  hA_csrOffsets = (cuI *)malloc((A_num_rows + 1) * sizeof(cuI));
  hA_columns = (cuI *)malloc(A_nnz * sizeof(cuI));
  hA_values = (cuZ *)malloc(A_nnz * sizeof(cuZ));
  hB = (cuZ *)malloc(std::pow(2, num_qubits) * sizeof(cuZ));
  hC = (cuZ *)malloc(std::pow(2, num_qubits) * sizeof(cuZ));
  for (int i = 0; i < S; i++) {
    hC[i] = {0.0, 0.0};
  }
  hC_result = (cuZ *)malloc(std::pow(2, num_qubits) * sizeof(cuZ));

  get_A_matrix(sblk, matrix, hA_csrOffsets, hA_columns, hA_values);

  //--------------------------------------------------------------------------
  // DEVICE MEMORY MANAGEMENT
  cuI *dA_csrOffsets, *dA_columns;
  cuZ *dA_values, *dB, *dC;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(
      cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(cuI)))
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(cuI)))
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(cuZ)))
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(cuZ)))
  CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(cuZ)))

  CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(cuI), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(cuI),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(cuZ),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, h_sv, B_size * sizeof(cuZ), cudaMemcpyHostToDevice))
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
                                   CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F))
  // Create dense matrix B
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                     CUDA_C_64F, order))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                     CUDA_C_64F, order))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, trans, trans, &alpha, matA,
                                         matB, &beta, matC, CUDA_C_64F, algo,
                                         &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute SpMM
  cudaDeviceSynchronize();
  double startTime = getTime();
  for (int c = 0; c < nrepeat; c++) {
    CHECK_CUSPARSE(cusparseSpMM(handle, trans, trans, &alpha, matA, matB, &beta,
                                matC, CUDA_C_64F, algo, dBuffer))
  }
  cudaDeviceSynchronize();
  auto time = (getTime() - startTime);

  double Gbytes = 1.0e-9 * double(sizeof(cuZ) * S);
  printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
         time, Gbytes * nrepeat / time);

  //--------------------------------------------------------------------------
  // VALIDATE

  CHECK_CUSPARSE(cusparseDnMatGetValues(matC, (void **)&dC))
  CHECK_CUDA(cudaMemcpy(hC, dC, C_size * sizeof(cuZ), cudaMemcpyDeviceToHost))
  bool correct = true;
  for (int i = 0; i < C_size; i++) {
    // printf("%f %f \n", hC[i].y, h_sv_result[i].y);
    if ((hC[i].x != h_sv_result[i].x) || (hC[i].y != h_sv_result[i].y)) {
      correct = false;
      // break;
    }
  }
  if (correct)
    printf("example PASSED\n");
  else
    printf("example FAILED: wrong result\n");

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
  return 0;
}

void apply_cuquantum(size_t num_qubits, size_t S, size_t T, cuZ *h_sv,
                     cuZ *h_sv_result, cuZ matrix[], size_t nrepeat) {
  const int nIndexBits = num_qubits;
  const int nSvSize = S;
  const int nTargets = 1;
  const int nControls = 0;
  const int adjoint = 0;

  int targets[] = {static_cast<int>(T)};
  int controls[] = {};

  cuZ *d_sv;
  cudaMalloc((void **)&d_sv, nSvSize * sizeof(cuZ));

  cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuZ), cudaMemcpyHostToDevice);

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

  cudaDeviceSynchronize();
  double startTime = getTime();
  for (int c = 0; c < nrepeat; c++) {
    custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, nIndexBits, matrix,
                          CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint,
                          targets, nTargets, controls, nullptr, nControls,
                          CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                          extraWorkspaceSizeInBytes);
  }
  cudaDeviceSynchronize();
  auto time = (getTime() - startTime);

  double Gbytes = 1.0e-9 * double(sizeof(cuZ) * S);
  printf("%12d, %12d, %12d, %12g, %12g, %12g\n", S, T, nrepeat, Gbytes * 1000,
         time, Gbytes * nrepeat / time);

  // destroy handle
  custatevecDestroy(handle);

  //--------------------------------------------------------------------------

  cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuZ), cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < nSvSize; i++) {
    if ((h_sv[i].x != h_sv_result[i].x) || (h_sv[i].y != h_sv_result[i].y)) {
      correct = false;
      break;
    }
  }

  if (correct)
    printf("example PASSED\n");
  else
    printf("example FAILED: wrong result\n");

  for (int i = 0; i < nSvSize; i++) {
    h_sv_result[i].x = h_sv[i].x;
    h_sv_result[i].y = h_sv[i].y;
  }

  cudaFree(d_sv);
  if (extraWorkspaceSizeInBytes)
    cudaFree(extraWorkspace);
}

void get_A_matrix(const int sblk, cuZ matrix[], cuI *hA_csrOffsets,
                  cuI *hA_columns, cuZ *hA_values) {
  std::size_t count = 0;
  std::size_t i;
  for (i = 0; i < sblk; ++i) {
    hA_csrOffsets[i] = count;
    hA_columns[count] = i;
    hA_values[count] = matrix[1];
    count += 1;
    hA_columns[count] = i + sblk;
    hA_values[count] = matrix[0];
    count += 1;
  }
  for (i = sblk; i < 2 * sblk; ++i) {
    hA_csrOffsets[i] = count;
    hA_columns[count] = i - sblk;
    hA_values[count] = matrix[3];
    count += 1;
    hA_columns[count] = i;
    hA_values[count] = matrix[2];
    count += 1;
  }
  hA_csrOffsets[++i] = count;
}

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}
