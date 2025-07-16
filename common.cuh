#pragma once
#include <functional>
#include <iostream>
#include <stdio.h>

inline void myAssert(bool condition, const char *msg, const char *file,
                     int line, int code = 1, bool abort = true) {

  if (!condition) {
    printf("Assert failed: %s %s %d\n", msg, file, line);
    if (abort)
      exit(code);
  }
}

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  myAssert(code == cudaSuccess, cudaGetErrorString(code), file, line, code,
           abort);
}

#define cudaCheck(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

#define gemmCheck(ans, msg)                                                    \
  {                                                                            \
    myAssert((ans), (msg), __FILE__, __LINE__);                                \
  }

void checkResult(const float *__restrict__ a, const float *__restrict__ b,
                 const float *__restrict__ c, size_t m, size_t k, size_t n,
                 bool ones) {

  auto gt = new float[m * n];
  if (!ones) {
    for (size_t row = 0; row < m; ++row) {
      for (size_t col = 0; col < n; ++col) {
        float c = 0.0;
        for (size_t i = 0; i < k; ++i) {
          c += a[row * k + i] * b[i * n + col];
        }
        gt[row * n + col] = c;
      }
    }
  } else {
    for (size_t i = 0; i < m * n; ++i)
      gt[i] = (float)k;
  }

  /// Check result
  double eps = 1.e-6; // machine zero
  for (int i = 0; i < m * n; ++i) {
    double abs_err = fabs(c[i] - gt[i]);
    if (abs_err > eps) {
      printf("Error! Matrix[%05df]=%.8f, abs_error=%.8f, ref=%.8f error term "
             "is > %E \n",
             i, c[i], abs_err, gt[i], eps);
      exit(1);
    }
  }
}

void profile(
    size_t m, size_t k, size_t n, int n_iter,
    std::function<void(const float *__restrict__ a, const float *__restrict__ b,
                       float *__restrict__ c, size_t m, size_t k, size_t n,
                       cudaStream_t stream, int n_iter)>
        launch_kernel,
    bool ones = true, bool print = false, const char *name = "") {

  float *a, *b, *c;
  auto a_bytes = m * k * sizeof(float);
  auto b_bytes = k * n * sizeof(float);
  auto c_bytes = m * n * sizeof(float);
  cudaCheck(cudaMallocHost(&a, a_bytes));
  cudaCheck(cudaMallocHost(&b, b_bytes));
  cudaCheck(cudaMallocHost(&c, c_bytes));

  /// Set memory
  for (size_t i = 0; i < m * k; ++i)
    a[i] = ones ? 1 : i;

  for (size_t i = 0; i < k * n; ++i)
    b[i] = ones ? 1 : i;
  cudaCheck(cudaMemset(c, 0, c_bytes));

  float *d_a, *d_b, *d_c;
  cudaCheck(cudaMalloc((void **)&d_a, a_bytes));
  cudaCheck(cudaMalloc((void **)&d_b, b_bytes));
  cudaCheck(cudaMalloc((void **)&d_c, c_bytes));

  cudaEvent_t start, stop;
  cudaStream_t stream;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaCheck(cudaMemcpyAsync(d_a, a, a_bytes, cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(d_b, b, b_bytes, cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemsetAsync(d_c, 0, c_bytes, stream));

  /// Wait for memory copy
  cudaCheck(cudaStreamSynchronize(stream));

  cudaCheck(cudaEventRecord(start, stream));
  launch_kernel(d_a, d_b, d_c, m, k, n, stream, n_iter);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaEventRecord(stop, stream));

  /// wait for the stop event complete
  cudaCheck(cudaEventSynchronize(stop));

  /// Metrics
  float msec_total = 0.0f;
  cudaCheck(cudaEventElapsedTime(&msec_total, start, stop));

  float msec_per_matmul = msec_total / n_iter;
  double flops_per_matmul = 2.0 * static_cast<double>(m) *
                            static_cast<double>(n) * static_cast<double>(k);
  double giga_flops =
      (flops_per_matmul * 1.0e-9f) / (msec_per_matmul / 1000.0f);

  printf("%s:  Total elapsed: %.3f msec. Performance= "
         "%.2f GFlop/s, Per matrix elapsed "
         "time= "
         "%.3f msec, Per "
         "matrix size= "
         "%.0f Ops\n",
         name, msec_total, giga_flops, msec_per_matmul, flops_per_matmul);

  cudaCheck(cudaMemcpyAsync(c, d_c, c_bytes, cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));

  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_c));

  cudaCheck(cudaStreamDestroy(stream));
  cudaCheck(cudaEventDestroy(stop));
  cudaCheck(cudaEventDestroy(start));

  if (print) {
    cudaDeviceSynchronize();
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        std::cout << c[i * n + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  checkResult(a, b, c, m, k, n, ones);

  cudaCheck(cudaFreeHost(a));
  cudaCheck(cudaFreeHost(b));
  cudaCheck(cudaFreeHost(c));
}
