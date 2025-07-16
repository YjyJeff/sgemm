#include "common.cuh"

/// Thread block size of the this kernel:
/// (M_BLOCK_SIZE / M_THREAD_TILING_SIZE, N_BLOCK_SIZE /N_THREAD_TILING_SIZE)
template <int M_BLOCK_SIZE, int K_BLOCK_SIZE, int N_BLOCK_SIZE,
          int M_THREAD_TILING_SIZE, int N_THREAD_TILING_SIZE>
__global__ void thread_tiling_matmul(const float *__restrict__ a,
                                     const float *__restrict__ b,
                                     float *__restrict__ c, int k, int n) {
  /// by * M_BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * M_BLOCK_SIZE;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * N_BLOCK_SIZE;

  constexpr auto num_threads = (M_BLOCK_SIZE / M_THREAD_TILING_SIZE) *
                               (N_BLOCK_SIZE / N_THREAD_TILING_SIZE);
  constexpr auto a_sub_per_thread_load_num =
      M_BLOCK_SIZE * K_BLOCK_SIZE / num_threads;
  constexpr auto b_sub_per_thread_load_num =
      K_BLOCK_SIZE * N_BLOCK_SIZE / num_threads;
  auto tiling_thread_id =
      threadIdx.y * (N_BLOCK_SIZE / N_THREAD_TILING_SIZE) + threadIdx.x;

  /// Declare shared variable
  __shared__ float a_sub[M_BLOCK_SIZE][K_BLOCK_SIZE];
  __shared__ float b_sub[K_BLOCK_SIZE][N_BLOCK_SIZE];

  float c_tiling[M_THREAD_TILING_SIZE][N_THREAD_TILING_SIZE] = {0.0f};

  auto num_iteration = k / K_BLOCK_SIZE;
  for (int iter = 0; iter < num_iteration; ++iter) {
    int a_sub_matrix_begin = a_block_begin + iter * K_BLOCK_SIZE;
    int b_sub_matrix_begin = b_block_begin + n * iter * K_BLOCK_SIZE;

    /// Load bm * bn and bn * bk into shared memory.
    for (int i = 0; i < a_sub_per_thread_load_num; ++i) {
      auto index_in_block = i * num_threads + tiling_thread_id;
      auto row_index = index_in_block / K_BLOCK_SIZE;
      auto col_index = index_in_block % K_BLOCK_SIZE;
      a_sub[row_index][col_index] =
          a[a_sub_matrix_begin + row_index * k + col_index];
    }

    for (int i = 0; i < b_sub_per_thread_load_num; ++i) {
      auto index_in_block = i * num_threads + tiling_thread_id;
      auto row_index = index_in_block / N_BLOCK_SIZE;
      auto col_index = index_in_block % N_BLOCK_SIZE;
      b_sub[row_index][col_index] =
          b[b_sub_matrix_begin + row_index * n + col_index];
    }

    /// confirm that the sub-matrix is loaded in the thread block
    __syncthreads();

    /// Compute with inner product
    for (int row = 0; row < M_THREAD_TILING_SIZE; ++row) {
      auto thread_tiling_row = threadIdx.y * M_THREAD_TILING_SIZE + row;
      for (int col = 0; col < N_THREAD_TILING_SIZE; ++col) {
        auto thread_tiling_col = threadIdx.x * N_THREAD_TILING_SIZE + col;
        for (int p = 0; p < K_BLOCK_SIZE; ++p)
          c_tiling[row][col] +=
              a_sub[thread_tiling_row][p] * b_sub[p][thread_tiling_col];
      }
    }

    /// all of the threads in the block finished the computation
    __syncthreads();
  }

  /// Set value
  auto c_sub_begin = n * blockIdx.y * M_BLOCK_SIZE + blockIdx.x * N_BLOCK_SIZE;
  auto c_tiling_begin = c_sub_begin + threadIdx.y * M_THREAD_TILING_SIZE * n +
                        threadIdx.x * N_THREAD_TILING_SIZE;

  for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
    for (int j = 0; j < N_THREAD_TILING_SIZE; ++j) {
      c[c_tiling_begin + i * n + j] = c_tiling[i][j];
    }
  }
}

/// Thread block size of the this kernel:
/// (M_BLOCK_SIZE / M_THREAD_TILING_SIZE, N_BLOCK_SIZE / N_THREAD_TILING_SIZE)
template <int M_BLOCK_SIZE, int K_BLOCK_SIZE, int N_BLOCK_SIZE,
          int M_THREAD_TILING_SIZE, int N_THREAD_TILING_SIZE>
__global__ void reg_thread_tiling_matmul(const float *__restrict__ a,
                                         const float *__restrict__ b,
                                         float *__restrict__ c, int k, int n) {
  /// by * M_BLOCK_SIZE is the ith block in row dim
  int row_block_idx = blockIdx.y * M_BLOCK_SIZE;
  int a_block_begin = row_block_idx * k;

  /// b always start from the blockId.x
  int col_block_idx = blockIdx.x;
  int b_block_begin = col_block_idx * N_BLOCK_SIZE;

  constexpr auto num_threads = (M_BLOCK_SIZE / M_THREAD_TILING_SIZE) *
                               (N_BLOCK_SIZE / N_THREAD_TILING_SIZE);
  constexpr auto a_sub_per_thread_load_num =
      M_BLOCK_SIZE * K_BLOCK_SIZE / num_threads;
  constexpr auto b_sub_per_thread_load_num =
      K_BLOCK_SIZE * N_BLOCK_SIZE / num_threads;
  auto tiling_thread_id =
      threadIdx.y * (N_BLOCK_SIZE / N_THREAD_TILING_SIZE) + threadIdx.x;

  /// Declare shared variable
  __shared__ float a_sub[M_BLOCK_SIZE][K_BLOCK_SIZE];
  __shared__ float b_sub[K_BLOCK_SIZE][N_BLOCK_SIZE];

  float c_tiling[M_THREAD_TILING_SIZE][N_THREAD_TILING_SIZE] = {0.0f};

  auto num_iteration = k / K_BLOCK_SIZE;
  for (int iter = 0; iter < num_iteration; ++iter) {
    int a_sub_matrix_begin = a_block_begin + iter * K_BLOCK_SIZE;
    int b_sub_matrix_begin = b_block_begin + n * iter * K_BLOCK_SIZE;

    /// Load bm * bn and bn * bk into shared memory.
#pragma unroll
    for (int i = 0; i < a_sub_per_thread_load_num; ++i) {
      auto index_in_block = i * num_threads + tiling_thread_id;
      auto row_index = index_in_block / K_BLOCK_SIZE;
      auto col_index = index_in_block % K_BLOCK_SIZE;
      a_sub[row_index][col_index] =
          a[a_sub_matrix_begin + row_index * k + col_index];
    }
#pragma unroll
    for (int i = 0; i < b_sub_per_thread_load_num; ++i) {
      auto index_in_block = i * num_threads + tiling_thread_id;
      auto row_index = index_in_block / N_BLOCK_SIZE;
      auto col_index = index_in_block % N_BLOCK_SIZE;
      b_sub[row_index][col_index] =
          b[b_sub_matrix_begin + row_index * n + col_index];
    }

    /// confirm that the sub-matrix is loaded in the thread block
    __syncthreads();

    auto a_tiling_begin = threadIdx.y * M_THREAD_TILING_SIZE;
    auto b_tiling_begin = threadIdx.x * N_THREAD_TILING_SIZE;

    for (int p = 0; p < K_BLOCK_SIZE; ++p) {
      /// load shared memory to register
      float a_tiling[M_THREAD_TILING_SIZE] = {0.0f};
      for (int i = 0; i < M_THREAD_TILING_SIZE; ++i)
        a_tiling[i] = a_sub[a_tiling_begin + i][p];

      float b_tiling[N_THREAD_TILING_SIZE] = {0.0f};
      for (int i = 0; i < N_THREAD_TILING_SIZE; ++i)
        b_tiling[i] = b_sub[p][b_tiling_begin + i];

      /// Compute outer product
      for (int row = 0; row < M_THREAD_TILING_SIZE; ++row) {
        for (int col = 0; col < N_THREAD_TILING_SIZE; ++col) {
          c_tiling[row][col] += a_tiling[row] * b_tiling[col];
        }
      }
    }

    /// all of the threads in the block finished the computation
    __syncthreads();
  }

  /// Set value
  auto c_sub_begin = n * blockIdx.y * M_BLOCK_SIZE + blockIdx.x * N_BLOCK_SIZE;
  auto c_tiling_begin = c_sub_begin + threadIdx.y * M_THREAD_TILING_SIZE * n +
                        threadIdx.x * N_THREAD_TILING_SIZE;

  for (int i = 0; i < M_THREAD_TILING_SIZE; ++i) {
    for (int j = 0; j < N_THREAD_TILING_SIZE; ++j) {
      c[c_tiling_begin + i * n + j] = c_tiling[i][j];
    }
  }
}

constexpr int m = 1536;
constexpr int k = 8960;
constexpr int n = 1536;
constexpr int n_iter = 1000;

int main() {
  {

    auto launch_kernel = [](const float *__restrict__ a,
                            const float *__restrict__ b, float *__restrict__ c,
                            int m, int k, int n, cudaStream_t stream,
                            int n_iter) {
      constexpr int M_BLOCK_SIZE = 64;
      constexpr int N_BLOCK_SIZE = 64;
      /// It will affect the size of the shared memory
      constexpr int K_BLOCK_SIZE = 8;
      constexpr int M_THREAD_TILING_SIZE = 8;
      constexpr int N_THREAD_TILING_SIZE = 8;
      constexpr int NUM_THREADS = (M_BLOCK_SIZE / M_THREAD_TILING_SIZE) *
                                  (N_BLOCK_SIZE / N_THREAD_TILING_SIZE);
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
                "M_BLOCK_SIZE * K_BLOCK_SIZE must be mulpily of NUM_THREADS in "
                "block");
      gemmCheck(K_BLOCK_SIZE * N_BLOCK_SIZE % NUM_THREADS == 0,
                "K_BLOCK_SIZE * N_BLOCK_SIZE must be mulpily of NUM_THREADS in "
                "block");

      dim3 block_dim(N_BLOCK_SIZE / N_THREAD_TILING_SIZE,
                     M_BLOCK_SIZE / M_THREAD_TILING_SIZE);
      dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

      for (auto i = 0; i < n_iter; ++i)
        thread_tiling_matmul<M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE,
                             M_THREAD_TILING_SIZE, N_THREAD_TILING_SIZE>
            <<<grid_dim, block_dim, 0, stream>>>(a, b, c, k, n);
    };

    profile(m, k, n, n_iter, launch_kernel, true, false, "thread_tiling");
  }

  {
    auto launch_kernel = [](const float *__restrict__ a,
                            const float *__restrict__ b, float *__restrict__ c,
                            int m, int k, int n, cudaStream_t stream,
                            int n_iter) {
      constexpr int M_BLOCK_SIZE = 64;
      constexpr int N_BLOCK_SIZE = 64;
      /// It will affect the size of the shared memory
      constexpr int K_BLOCK_SIZE = 8;
      constexpr int M_THREAD_TILING_SIZE = 16;
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

      dim3 block_dim(N_BLOCK_SIZE / N_THREAD_TILING_SIZE,
                     M_BLOCK_SIZE / M_THREAD_TILING_SIZE);
      dim3 grid_dim(n / N_BLOCK_SIZE, m / M_BLOCK_SIZE);

      for (auto i = 0; i < n_iter; ++i)
        reg_thread_tiling_matmul<M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE,
                                 M_THREAD_TILING_SIZE, N_THREAD_TILING_SIZE>
            <<<grid_dim, block_dim, 0, stream>>>(a, b, c, k, n);
    };

    profile(m, k, n, n_iter, launch_kernel, true, false,
            "out_product_thread_tiling");
  }
}
