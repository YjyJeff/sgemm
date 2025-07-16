#include "common.cuh"

__global__ void blocked_gemm(const float *__restrict__ a,
                             const float *__restrict__ b, float *__restrict__ c,
                             size_t m, size_t k, size_t n) {

  auto m_block_size = blockDim.y;
  auto n_block_size = blockDim.x;

  /// by * BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * m_block_size;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * n_block_size;

  float c_sub = 0;
  for (int i = 0; i < k; ++i) {
    c_sub += a[a_block_begin + threadIdx.y * k + i] *
             b[b_block_begin + i * n + threadIdx.x];
  }
  /// Set value
  int c_sub_begin = n * blockIdx.y * m_block_size + blockIdx.x * n_block_size;
  c[c_sub_begin + threadIdx.y * n + threadIdx.x] = c_sub;
}

template <size_t K_BLOCK_SIZE>
__global__ void k_tiling_blocked_matmul(const float *__restrict__ a,
                                        const float *__restrict__ b,
                                        float *__restrict__ c, int k, int n) {
  auto m_block_size = blockDim.y;
  auto n_block_size = blockDim.x;

  /// by * BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * m_block_size;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * n_block_size;

  float Csub = 0;
  auto num_iteration = k / K_BLOCK_SIZE;
  for (int i = 0; i < num_iteration; ++i) {
    int a_sub_matrix_begin = a_block_begin + i * K_BLOCK_SIZE;
    int b_sub_matrix_begin = b_block_begin + n * i * K_BLOCK_SIZE;

#pragma unroll
    for (int j = 0; j < K_BLOCK_SIZE; ++j)
      Csub += a[a_sub_matrix_begin + threadIdx.y * k + j] *
              b[b_sub_matrix_begin + j * n + threadIdx.x];
  }

  /// Set value
  int c_sub_begin = n * blockIdx.y * m_block_size + blockIdx.x * n_block_size;
  c[c_sub_begin + threadIdx.y * n + threadIdx.x] = Csub;
}

int main() {

  size_t m = 1536;
  size_t n = 8960;
  size_t k = 1536;
  int n_iter = 10;

  /// naive
  {
    auto launch_kernel = [](const float *__restrict__ a,
                            const float *__restrict__ b, float *__restrict__ c,
                            size_t m, size_t k, size_t n, cudaStream_t stream,
                            int n_iter) {
      const size_t M_BLOCK_SIZE = 32;
      const size_t N_BLOCK_SIZE = 32;
      gemmCheck(m >= M_BLOCK_SIZE && m % M_BLOCK_SIZE == 0,
                "M must be multiply of M_BLOCK_SIZE");
      gemmCheck(n >= N_BLOCK_SIZE && n % N_BLOCK_SIZE == 0,
                "K must be multiply of K_BLOCK_SIZE");

      dim3 block_dim(N_BLOCK_SIZE, M_BLOCK_SIZE);
      dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

      for (auto i = 0; i < n_iter; ++i)
        blocked_gemm<<<grid_dim, block_dim, 0, stream>>>(a, b, c, m, k, n);
    };

    profile(m, n, k, n_iter, launch_kernel, true, false, "blocked");
  }

  /// k tiling memory access
  {
    auto launch_kernel = [](const float *__restrict__ a,
                            const float *__restrict__ b, float *__restrict__ c,
                            size_t m, size_t k, size_t n, cudaStream_t stream,
                            int n_iter) {
      const size_t M_BLOCK_SIZE = 32;
      const size_t N_BLOCK_SIZE = 32;
      /// It will affect the size of the shared memory
      const size_t K_BLOCK_SIZE = 8;
      gemmCheck(m >= M_BLOCK_SIZE && m % M_BLOCK_SIZE == 0,
                "M must be multiply of M_BLOCK_SIZE");
      gemmCheck(k >= K_BLOCK_SIZE && k % K_BLOCK_SIZE == 0,
                "K must be multiply of K_BLOCK_SIZE");
      gemmCheck(n >= N_BLOCK_SIZE && n % N_BLOCK_SIZE == 0,
                "N must be multiply of N_BLOCK_SIZE");

      dim3 block_dim(N_BLOCK_SIZE, M_BLOCK_SIZE);
      dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

      for (auto i = 0; i < n_iter; ++i)
        k_tiling_blocked_matmul<K_BLOCK_SIZE>
            <<<grid_dim, block_dim, 0, stream>>>(a, b, c, k, n);
    };

    profile(m, k, n, n_iter, launch_kernel, true, false, "k_tiling_blocked");
  }
}
