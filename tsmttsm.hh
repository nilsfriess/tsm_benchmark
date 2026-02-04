#pragma once

#include <sycl/sycl.hpp>

/* This file contains several implementations of a tall-skinny times tall-skinny matrix
   matrix product A^T * B = C, where A and B are a tall-skinny KxM matrices (i.e. K >> M)
   and C (the output matrix) is of size MxM.

   All matrices are assumed to be stored in row-major order.
*/

/* Variant 1: Parallelises only over K using a grid-stride loop.
   The stride can be configured using a template parameter.
*/
template <typename T, int M_, int stride>
sycl::event tsmttsm(sycl::queue &q, int K, T* A, T* B, T* C) {
  constexpr auto M = M_;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[M][M] = {0};

      for (int k=item; k < K; k += stride) {
        for (int m=0; m < M; ++m)
          for (int n=0; n < M; ++n)
            c_local[m][n] += A[k * M + m] * B[k * M + n];
      }

      for (int m=0; m < M; ++m)
        for (int n=0; n < M; ++n) {
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[m * M + n]);
          C_ref += c_local[m][n];
        }
    });
  });
}

/* Variant 2: Parallelises over K using a grid-stride loop and uses tiling.
   The max tile size and the stride can be configured using a template parameter.
*/
template <typename T, int M, int stride, int max_tile_size>
sycl::event tsmttsm2(sycl::queue &q, int K, T* A, T* B, T* C) {
  constexpr auto tile_size = std::min(max_tile_size, M); // tiles are max. 4x4
  static_assert((M*M) % (tile_size * tile_size) == 0, "Currently it is assumed that the tiles fit perfectly");
  constexpr auto num_tiles = (M*M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;
  
  return q.submit([&](sycl::handler &cgh) { 
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int tid = item[0];

      auto k_start = tid / num_tiles;
      auto k_stride = stride / num_tiles;

      auto tile_idx = tid % num_tiles;
      auto tile_row_start = (tile_idx / tiles_per_dim) * tile_size;
      auto tile_col_start = (tile_idx % tiles_per_dim) * tile_size;

      for (int k=k_start; k < K; k += k_stride) {
        for (int m=0; m < tile_size; ++m)
          for (int n=0; n < tile_size; ++n)
            c_local[m][n] += A[k * M + (tile_row_start + m)] * B[k * M + (tile_col_start + n)];
      }

      for (int m=0; m < tile_size; ++m)
        for (int n=0; n < tile_size; ++n) {
          auto gidx = (tile_row_start + m) * M + (tile_col_start + n);
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
          C_ref += c_local[m][n];
        }
    });
  });
}

/* Variant 3: Same as variant 2 but explicitly loads the required values of A and B into registers.

   This shouldn't really make a difference, the compiler should be smart enough to do this.
*/
template <typename T, int M, int stride, int max_tile_size>
sycl::event tsmttsm3(sycl::queue &q, int K, T* A, T* B, T* C) {
  constexpr auto tile_size = std::min(max_tile_size, M); // tiles are max. 4x4
  static_assert((M*M) % (tile_size * tile_size) == 0, "Currently it is assumed that the tiles fit perfectly");
  constexpr auto num_tiles = (M*M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;
  
  return q.submit([&](sycl::handler &cgh) { 
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int tid = item[0];

      auto k_start = tid / num_tiles;
      auto k_stride = stride / num_tiles;

      auto tile_idx = tid % num_tiles;
      auto tile_row_start = (tile_idx / tiles_per_dim) * tile_size;
      auto tile_col_start = (tile_idx % tiles_per_dim) * tile_size;

      T A_vals[tile_size];
      T B_vals[tile_size];        
      
      for (int k=k_start; k < K; k += k_stride) {
        for (int i=0; i<tile_size; ++i) {
          A_vals[i] = A[k * M + (tile_row_start + i)];
          B_vals[i] = B[k * M + (tile_col_start + i)];
        }
        
        for (int m=0; m < tile_size; ++m)
          for (int n=0; n < tile_size; ++n)
            c_local[m][n] += A_vals[m] * B_vals[n];
      }

      for (int m=0; m < tile_size; ++m)
        for (int n=0; n < tile_size; ++n) {
          auto gidx = (tile_row_start + m) * M + (tile_col_start + n);
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
          C_ref += c_local[m][n];
        }
    });
  });
}

/* Variant 4: Same as variant 2 but the tiles are "transposed". */
template <typename T, int M, int stride, int max_tile_size>
sycl::event tsmttsm4(sycl::queue& q, int K, T* A, T* B, T* C)
{
  constexpr auto tile_size = std::min(max_tile_size, M); // tiles are max. 4x4
  static_assert((M * M) % (tile_size * tile_size) == 0, "Currently it is assumed that the tiles fit perfectly");
  constexpr auto num_tiles = (M * M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;

  return q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int tid = item[0];
      auto k_start = tid / num_tiles;
      auto k_stride = stride / num_tiles;

      auto tile_idx = tid % num_tiles;
      auto tile_row_start = (tile_idx % tiles_per_dim);
      auto tile_col_start = (tile_idx / tiles_per_dim);

      T A_vals[tile_size];
      T B_vals[tile_size];

      for (int k = k_start; k < K; k += k_stride) {
        for (int i = 0; i < tile_size; ++i) {
          A_vals[i] = A[k * M + (tile_row_start + i * tiles_per_dim)];
          B_vals[i] = B[k * M + (tile_col_start + i * tiles_per_dim)];
        }

        for (int m = 0; m < tile_size; ++m)
          for (int n = 0; n < tile_size; ++n) c_local[m][n] += A_vals[m] * B_vals[n];
      }

      for (int m = 0; m < tile_size; ++m)
        for (int n = 0; n < tile_size; ++n) {
          auto gidx = (tile_row_start + m * tiles_per_dim) * M + (tile_col_start + n * tiles_per_dim);
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
          C_ref += c_local[m][n];
        }
    });
  });
}

/* Variant 5: Same as variant 3 but with double buffering of the A and B values (aka "leap frogging")
 */
template <typename T, int M, int stride, int max_tile_size>
sycl::event tsmttsm5(sycl::queue &q, int K, T* A, T* B, T* C) {
  constexpr auto tile_size = std::min(max_tile_size, M); // tiles are max. 4x4
  static_assert((M*M) % (tile_size * tile_size) == 0, "Currently it is assumed that the tiles fit perfectly");
  constexpr auto num_tiles = (M*M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;
  
  return q.submit([&](sycl::handler &cgh) { 
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int tid = item[0];

      auto k_start = tid / num_tiles;
      auto k_stride = stride / num_tiles;

      auto tile_idx = tid % num_tiles;
      auto tile_row_start = (tile_idx / tiles_per_dim) * tile_size;
      auto tile_col_start = (tile_idx % tiles_per_dim) * tile_size;

      T A_vals_curr[tile_size];
      T B_vals_curr[tile_size];

      for (int i=0; i<tile_size; ++i) {
        A_vals_curr[i] = A[k_start * M + (tile_row_start + i)];
        B_vals_curr[i] = B[k_start * M + (tile_col_start + i)];
      }

      T A_vals_next[tile_size];
      T B_vals_next[tile_size];
      
      for (int k=k_start; k < K; k += k_stride) {
        if (k + k_stride < K) {
          for (int i=0; i<tile_size; ++i) {
            A_vals_next[i] = A[(k + k_stride) * M + (tile_row_start + i)];
            B_vals_next[i] = B[(k + k_stride) * M + (tile_col_start + i)];
          }
        }
        
        for (int m=0; m < tile_size; ++m)
          for (int n=0; n < tile_size; ++n)
            c_local[m][n] += A_vals_curr[m] * B_vals_curr[n];

        std::swap(A_vals_next, A_vals_curr);
        std::swap(B_vals_next, B_vals_curr);        
      }

      for (int m=0; m < tile_size; ++m)
        for (int n=0; n < tile_size; ++n) {
          auto gidx = (tile_row_start + m) * M + (tile_col_start + n);
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
          C_ref += c_local[m][n];
        }
    });
  });
}

/* Variant 6: Similar to variant 5 but now we reduce in two stages: first within the work group, then the first thread
   within the work group reduces globally.
*/
template <typename T, int M, int global_size, int local_size, int max_tile_size>
sycl::event tsmttsm6(sycl::queue &q, int K, T* A, T* B, T* C) {
  constexpr auto tile_size = std::min(max_tile_size, M);
  static_assert((M*M) % (tile_size * tile_size) == 0);
  constexpr auto num_tiles = (M*M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;
  constexpr auto num_groups = global_size / local_size;
  
  return q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 2> C_shared({tile_size, tile_size}, cgh);
    
    cgh.parallel_for(sycl::nd_range<1>{global_size, local_size}, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int lid = item.get_local_id(0);
      int group_id = item.get_group(0);

      // All threads in work-group work on the SAME tile
      auto tile_idx = group_id % num_tiles;
      auto tile_row_start = (tile_idx / tiles_per_dim) * tile_size;
      auto tile_col_start = (tile_idx % tiles_per_dim) * tile_size;

      // Divide k iterations among threads in the work-group
      auto k_start = (group_id / num_tiles) * local_size + lid;
      auto k_stride = (num_groups / num_tiles) * local_size;

      if (lid == 0) {
        for (int m = 0; m < tile_size; ++m)
          for (int n = 0; n < tile_size; ++n)
            C_shared[m][n] = 0;
      }
      item.barrier(sycl::access::fence_space::local_space);

      T A_vals[tile_size];
      T B_vals[tile_size];

      for (int k = k_start; k < K; k += k_stride) {
        for (int i = 0; i < tile_size; ++i) {
          A_vals[i] = A[k * M + (tile_row_start + i)];
          B_vals[i] = B[k * M + (tile_col_start + i)];
        }

        for (int m = 0; m < tile_size; ++m)
          for (int n = 0; n < tile_size; ++n)
            c_local[m][n] += A_vals[m] * B_vals[n];
      }

      // Local reduction
      item.barrier(sycl::access::fence_space::local_space);
      for (int m = 0; m < tile_size; ++m)
        for (int n = 0; n < tile_size; ++n) {
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> C_ref(C_shared[m][n]);
          C_ref += c_local[m][n];
        }

      // Barrier before reading C_shared
      item.barrier(sycl::access::fence_space::local_space);

      if (lid == 0) {
        for (int m = 0; m < tile_size; ++m)
          for (int n = 0; n < tile_size; ++n) {
            auto gidx = (tile_row_start + m) * M + (tile_col_start + n);
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
            C_ref += C_shared[m][n];
          }
      }
    });
  });
}

/* Variant 7: Combines variant 4's transposed tiling with correct double buffering using ping-pong pattern.
   The double buffering allows the compiler to overlap memory loads with computation.
*/
template <typename T, int M, int stride, int max_tile_size>
sycl::event tsmttsm7(sycl::queue& q, int K, T* A, T* B, T* C)
{
  constexpr auto tile_size = std::min(max_tile_size, M);
  static_assert((M * M) % (tile_size * tile_size) == 0, "Currently it is assumed that the tiles fit perfectly");
  constexpr auto num_tiles = (M * M) / (tile_size * tile_size);
  constexpr auto tiles_per_dim = M / tile_size;

  return q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(stride, [=](auto item) {
      T c_local[tile_size][tile_size] = {0};

      int tid = item[0];
      auto k_start = tid / num_tiles;
      auto k_stride = stride / num_tiles;

      auto tile_idx = tid % num_tiles;
      auto tile_row_start = (tile_idx % tiles_per_dim);
      auto tile_col_start = (tile_idx / tiles_per_dim);

      // Double buffering: two sets of buffers for ping-pong
      T A_vals[2][tile_size];
      T B_vals[2][tile_size];

      if (k_start < K) {
        // Preload first iteration into buffer 0
        for (int i = 0; i < tile_size; ++i) {
          A_vals[0][i] = A[k_start * M + (tile_row_start + i * tiles_per_dim)];
          B_vals[0][i] = B[k_start * M + (tile_col_start + i * tiles_per_dim)];
        }
      }

      int current_buf = 0;
      for (int k = k_start; k < K; k += k_stride) {
        int next_buf = 1 - current_buf;
        
        // Prefetch next iteration into the other buffer
        if (k + k_stride < K) {
          for (int i = 0; i < tile_size; ++i) {
            A_vals[next_buf][i] = A[(k + k_stride) * M + (tile_row_start + i * tiles_per_dim)];
            B_vals[next_buf][i] = B[(k + k_stride) * M + (tile_col_start + i * tiles_per_dim)];
          }
        }

        // Compute using current buffer - compiler can overlap this with above loads
        for (int m = 0; m < tile_size; ++m)
          for (int n = 0; n < tile_size; ++n) 
            c_local[m][n] += A_vals[current_buf][m] * B_vals[current_buf][n];

        current_buf = next_buf;
      }

      for (int m = 0; m < tile_size; ++m)
        for (int n = 0; n < tile_size; ++n) {
          auto gidx = (tile_row_start + m * tiles_per_dim) * M + (tile_col_start + n * tiles_per_dim);
          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> C_ref(C[gidx]);
          C_ref += c_local[m][n];
        }
    });
  });
}
