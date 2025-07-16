#include "common.cuh"

/// Thread block size of the this kernel:
/// (M_BLOCK_SIZE / M_THREAD_TILING_SIZE, N_BLOCK_SIZE / N_THREAD_TILING_SIZE)
template <int M_BLOCK_SIZE, int K_BLOCK_SIZE, int N_BLOCK_SIZE,
          int M_THREAD_TILING_SIZE, int N_THREAD_TILING_SIZE>
__global__ void prefetch_matmul(const float *__restrict__ a,
                                const float *__restrict__ b,
                                float *__restrict__ c, int k, int n) {
  /// by * M_BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * M_BLOCK_SIZE;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * N_BLOCK_SIZE;

  constexpr int NUM_THREADS = (M_BLOCK_SIZE / M_THREAD_TILING_SIZE) *
                              (N_BLOCK_SIZE / N_THREAD_TILING_SIZE);
  int tiling_thread_id =
      threadIdx.y * (N_BLOCK_SIZE / N_THREAD_TILING_SIZE) + threadIdx.x;

/// Declare shared variable with double buffer for prefetching the globla
/// memory
#ifdef TRANSPOSE
  __shared__ float a_sub[2][K_BLOCK_SIZE][M_BLOCK_SIZE];
#else
  __shared__ float a_sub[2][M_BLOCK_SIZE][K_BLOCK_SIZE];
#endif
  __shared__ float b_sub[2][K_BLOCK_SIZE][N_BLOCK_SIZE];
  /// Declare registers for prefetch the global memory
  constexpr int A_SUB_PER_THREAD_LOAD_NUM =
      M_BLOCK_SIZE * K_BLOCK_SIZE / NUM_THREADS;
  constexpr int B_SUB_PER_THREAD_LOAD_NUM =
      K_BLOCK_SIZE * N_BLOCK_SIZE / NUM_THREADS;
  float a_prefetch_regs[A_SUB_PER_THREAD_LOAD_NUM] = {0.0f};
  float b_prefetch_regs[B_SUB_PER_THREAD_LOAD_NUM] = {0.0f};
  /// Declare double buffer for prefetching the shared memory
  float a_tiling[2][M_THREAD_TILING_SIZE] = {0.0f};
  float b_tiling[2][N_THREAD_TILING_SIZE] = {0.0f};
  float c_tiling[M_THREAD_TILING_SIZE][N_THREAD_TILING_SIZE] = {0.0f};

  int a_tiling_begin = threadIdx.y * M_THREAD_TILING_SIZE;
  int b_tiling_begin = threadIdx.x * N_THREAD_TILING_SIZE;

  int a_col_index = tiling_thread_id % K_BLOCK_SIZE;
  int b_col_index = tiling_thread_id % N_BLOCK_SIZE;

/// Load the first blocked data to shared memory
#pragma unroll
  for (int i = 0; i < A_SUB_PER_THREAD_LOAD_NUM; ++i) {
    int index_in_block = i * NUM_THREADS + tiling_thread_id;
    int row_index = index_in_block / K_BLOCK_SIZE;
#ifdef TRANSPOSE
    a_sub[0][a_col_index][row_index] =
        a[a_block_begin + row_index * k + a_col_index];
#else
    a_sub[0][row_index][a_col_index] =
        a[a_block_begin + row_index * k + a_col_index];
#endif
  }
#pragma unroll
  for (int i = 0; i < B_SUB_PER_THREAD_LOAD_NUM; ++i) {
    int index_in_block = i * NUM_THREADS + tiling_thread_id;
    int row_index = index_in_block / N_BLOCK_SIZE;
    b_sub[0][row_index][b_col_index] =
        b[b_block_begin + row_index * n + b_col_index];
  }

  __syncthreads();

  /// load the first thread tile from shared memory
  for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
#ifdef TRANSPOSE
    a_tiling[0][i] = a_sub[0][0][a_tiling_begin + i];
#else
    a_tiling[0][i] = a_sub[0][a_tiling_begin + i][0];
#endif
  }

  for (int i = 0; i < N_THREAD_TILING_SIZE; ++i)
    b_tiling[0][i] = b_sub[0][0][b_tiling_begin + i];

  /// big iteration
  auto num_iteration = k / K_BLOCK_SIZE;
  for (int iter = 0; iter < num_iteration; ++iter) {
    /// Prefetch global memory that used in the next iteration into the
    /// registers
    auto read_idx = iter & 1;
    auto write_idx = 1 - read_idx;

    if (iter != num_iteration - 1) {
      auto next_a_sub_matrix_begin = a_block_begin + (iter + 1) * K_BLOCK_SIZE;
      auto next_b_sub_matrix_begin =
          b_block_begin + n * (iter + 1) * K_BLOCK_SIZE;
#pragma unroll
      for (int i = 0; i < A_SUB_PER_THREAD_LOAD_NUM; ++i) {
        auto index_in_block = i * NUM_THREADS + tiling_thread_id;
        auto row_index = index_in_block / K_BLOCK_SIZE;
        a_prefetch_regs[i] =
            a[next_a_sub_matrix_begin + row_index * k + a_col_index];
      }

#pragma unroll
      for (int i = 0; i < B_SUB_PER_THREAD_LOAD_NUM; ++i) {
        auto index_in_block = i * NUM_THREADS + tiling_thread_id;
        auto row_index = index_in_block / N_BLOCK_SIZE;
        b_prefetch_regs[i] =
            b[next_b_sub_matrix_begin + row_index * n + b_col_index];
      }
    }

#pragma unroll
    for (int p = 0; p < K_BLOCK_SIZE; ++p) {
      auto thread_tiling_read_idx = p & 1;
      auto thread_tiling_write_idx = 1 - thread_tiling_read_idx;
      if (p != K_BLOCK_SIZE - 1) {
        /// Load next thread tiling from shared memory to registers
#pragma unroll
        for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
#ifdef TRANSPOSE
          a_tiling[thread_tiling_write_idx][i] =
              a_sub[read_idx][p + 1][a_tiling_begin + i];
#else
          a_tiling[thread_tiling_write_idx][i] =
              a_sub[read_idx][a_tiling_begin + i][p + 1];
#endif
        }

#pragma unroll
        for (int i = 0; i < N_THREAD_TILING_SIZE; ++i)
          b_tiling[thread_tiling_write_idx][i] =
              b_sub[read_idx][p + 1][b_tiling_begin + i];
      }

      /// Compute outer product
#pragma unroll
      for (int row = 0; row < M_THREAD_TILING_SIZE; ++row) {
#pragma unroll
        for (int col = 0; col < N_THREAD_TILING_SIZE; ++col) {
          c_tiling[row][col] += a_tiling[thread_tiling_read_idx][row] *
                                b_tiling[thread_tiling_read_idx][col];
        }
      }
    }

    /// Store prefetched registers to shared memory
    if (iter != num_iteration - 1) {
#pragma unroll
      for (int i = 0; i < A_SUB_PER_THREAD_LOAD_NUM; ++i) {
        auto index_in_block = i * NUM_THREADS + tiling_thread_id;
        auto row_index = index_in_block / K_BLOCK_SIZE;
#ifdef TRANSPOSE
        a_sub[write_idx][a_col_index][row_index] = a_prefetch_regs[i];
#else
        a_sub[write_idx][row_index][a_col_index] = a_prefetch_regs[i];
#endif
      }
#pragma unroll
      for (int i = 0; i < B_SUB_PER_THREAD_LOAD_NUM; ++i) {
        auto index_in_block = i * NUM_THREADS + tiling_thread_id;
        auto row_index = index_in_block / N_BLOCK_SIZE;
        b_sub[write_idx][row_index][b_col_index] = b_prefetch_regs[i];
      }

      /// all of the registers has written to shared memory
      __syncthreads();
    }

    /// load the first thread tile from shared memory
#pragma unroll
    for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
#ifdef TRANSPOSE
      a_tiling[0][i] = a_sub[write_idx][0][a_tiling_begin + i];
#else
      a_tiling[0][i] = a_sub[write_idx][a_tiling_begin + i][0];
#endif
    }

#pragma unroll
    for (int i = 0; i < N_THREAD_TILING_SIZE; ++i)
      b_tiling[0][i] = b_sub[write_idx][0][b_tiling_begin + i];
  }

  /// Set value
  auto c_sub_begin = n * blockIdx.y * M_BLOCK_SIZE + blockIdx.x * N_BLOCK_SIZE;
  auto c_tiling_begin = c_sub_begin + threadIdx.y * M_THREAD_TILING_SIZE * n +
                        threadIdx.x * N_THREAD_TILING_SIZE;

#pragma unroll
  for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
#pragma unroll
    for (int j = 0; j < N_THREAD_TILING_SIZE; ++j) {
      c[c_tiling_begin + i * n + j] = c_tiling[i][j];
    }
  }
}

constexpr size_t m = 1536;
constexpr size_t k = 8960;
constexpr size_t n = 1536;
constexpr int n_iter = 1000;

int main() {

  auto launch_kernel = [](const float *__restrict__ a,
                          const float *__restrict__ b, float *__restrict__ c,
                          int m, int k, int n, cudaStream_t stream,
                          int n_iter) {
    constexpr int M_BLOCK_SIZE = 128;
    constexpr int N_BLOCK_SIZE = 128;
    /// It will affect the size of the shared memory
    constexpr int K_BLOCK_SIZE = 8;
    constexpr int M_THREAD_TILING_SIZE = 8;
    constexpr int N_THREAD_TILING_SIZE = 8;
    constexpr int NUM_THREADS = M_THREAD_TILING_SIZE * N_THREAD_TILING_SIZE;
    gemmCheck(m >= M_BLOCK_SIZE && m % M_BLOCK_SIZE == 0,
              "M must be multiply of M_BLOCK_SIZE");
    gemmCheck(k >= K_BLOCK_SIZE && k % K_BLOCK_SIZE == 0,
              "K must be multiply of K_BLOCK_SIZE");
    gemmCheck(n >= N_BLOCK_SIZE && n % N_BLOCK_SIZE == 0,
              "N must be multiply of N_BLOCK_SIZE");
    gemmCheck(M_BLOCK_SIZE >= M_THREAD_TILING_SIZE &&
                  M_THREAD_TILING_SIZE % M_THREAD_TILING_SIZE == 0,
              "M_BLOCK_SIZE must be multiply M_THREAD_TILING_SIZE");
    gemmCheck(N_BLOCK_SIZE >= N_THREAD_TILING_SIZE &&
                  N_BLOCK_SIZE % N_THREAD_TILING_SIZE == 0,
              "N_BLOCK_SIZE must be multiply of N_THREAD_TILING_SIZE");

    gemmCheck(M_BLOCK_SIZE * K_BLOCK_SIZE % NUM_THREADS == 0,
              "M_BLOCK_SIZE * K_BLOCK_SIZE must be mulpily of NUM_THREADS "
              "in block");
    gemmCheck(K_BLOCK_SIZE * N_BLOCK_SIZE % NUM_THREADS == 0,
              "K_BLOCK_SIZE * N_BLOCK_SIZE must be mulpily of NUM_THREADS in "
              "block");
    gemmCheck(NUM_THREADS >= K_BLOCK_SIZE && NUM_THREADS % K_BLOCK_SIZE == 0,
              "NUM_THREADS must be mulpily of K_BLOCK_SIZE");

    dim3 block_dim(N_BLOCK_SIZE / N_THREAD_TILING_SIZE,
                   M_BLOCK_SIZE / M_THREAD_TILING_SIZE);
    dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

    for (auto i = 0; i < n_iter; ++i)
      prefetch_matmul<M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE,
                      M_THREAD_TILING_SIZE, N_THREAD_TILING_SIZE>
          <<<grid_dim, block_dim, 0, stream>>>(a, b, c, k, n);
  };

  profile(m, k, n, n_iter, launch_kernel, true, false, "prefetch");
}
