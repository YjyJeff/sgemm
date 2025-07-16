#include "common.cuh"

template <size_t M_BLOCK_SIZE, size_t K_BLOCK_SIZE, size_t N_BLOCK_SIZE>
__global__ void shared_blocked_matmul(const float *__restrict__ a,
                                      const float *__restrict__ b,
                                      float *__restrict__ c, int k, int n) {
  /// by * BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * M_BLOCK_SIZE;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * N_BLOCK_SIZE;

  /// Declare shared variable
  __shared__ float a_sub[M_BLOCK_SIZE][K_BLOCK_SIZE];
  __shared__ float b_sub[K_BLOCK_SIZE][N_BLOCK_SIZE];

  float Csub = 0;
  auto num_iteration = k / K_BLOCK_SIZE;
  for (int i = 0; i < num_iteration; ++i) {

    int a_sub_matrix_begin = a_block_begin + i * K_BLOCK_SIZE;
    int b_sub_matrix_begin = b_block_begin + n * i * K_BLOCK_SIZE;

    /// Load bm * bn and bn * bk into shared memory.
    /// What if bm != bn != bk, how to load the memory?
    a_sub[threadIdx.y][threadIdx.x] =
        a[a_sub_matrix_begin + threadIdx.y * k + threadIdx.x];
    b_sub[threadIdx.y][threadIdx.x] =
        b[b_sub_matrix_begin + threadIdx.y * n + threadIdx.x];

    /// confirm that the sub-matrix is loaded in the thread block
    __syncthreads();

    ///
#pragma unroll
    for (int j = 0; j < K_BLOCK_SIZE; ++j)
      Csub += a_sub[threadIdx.y][j] * b_sub[j][threadIdx.x];

    /// all of the threads in the block finished the computation
    __syncthreads();
  }

  /// Set value
  int c_sub_begin = n * blockIdx.y * M_BLOCK_SIZE + blockIdx.x * N_BLOCK_SIZE;
  c[c_sub_begin + threadIdx.y * n + threadIdx.x] = Csub;
}

int main() {

  size_t m = 1536;
  size_t k = 8960;
  size_t n = 1536;
  int n_iter = 100;

  auto launch_kernel = [](const float *__restrict__ a,
                          const float *__restrict__ b, float *__restrict__ c,
                          size_t m, size_t k, size_t n, cudaStream_t stream,
                          int n_iter) {
    const size_t M_BLOCK_SIZE = 32;
    const size_t N_BLOCK_SIZE = 32;
    /// It will affect the size of the shared memory
    const size_t K_BLOCK_SIZE = 32;
    gemmCheck(m >= M_BLOCK_SIZE && m % M_BLOCK_SIZE == 0,
              "M must be multiply of M_BLOCK_SIZE");
    gemmCheck(k >= K_BLOCK_SIZE && k % K_BLOCK_SIZE == 0,
              "K must be multiply of K_BLOCK_SIZE");
    gemmCheck(n >= N_BLOCK_SIZE && n % N_BLOCK_SIZE == 0,
              "N must be multiply of N_BLOCK_SIZE");

    dim3 block_dim(N_BLOCK_SIZE, M_BLOCK_SIZE);
    dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

    for (auto i = 0; i < n_iter; ++i)
      shared_blocked_matmul<M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE>
          <<<grid_dim, block_dim, 0, stream>>>(a, b, c, k, n);
  };

  profile(m, k, n, n_iter, launch_kernel, true, false, "shared_blocked");
}
