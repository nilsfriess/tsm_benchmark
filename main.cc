#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <sycl/sycl.hpp>
#include "tsmttsm.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
std::vector<T> compute_naive(int K, int M, T* A, T* B) {
  std::vector<T> C(M * M, 0);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < M; ++n) {
      for (int k = 0; k < K; ++k) {
        C[m * M + n] += A[k * M + m] * B[k * M + n];
      }
    }
  }

  return C;
}

template <class T>
bool is_correct(const std::vector<T>& C_correct, T* C, int M, T rel_tol = 1e-6, T abs_tol = 1e-10) {
  for (int i = 0; i < M * M; ++i) {
    T diff = std::abs(C_correct[i] - C[i]);
    T magnitude = std::max(std::abs(C_correct[i]), std::abs(C[i]));
    if (diff > abs_tol && diff > rel_tol * magnitude) {
      std::cerr << "Mismatch at index " << i << ": expected " << C_correct[i]
                << ", got " << C[i] << " (diff=" << diff << ")\n";
      return false;
    }
  }
  return true;
}

struct BenchmarkStats {
  double avg_ns;
  double min_ns;
  double max_ns;
  double stddev_ns;
  double gflops;
};

template <typename Scalar, int M, int K, int stride, int local_size, int max_tile_size, int warmup, int runs>
void run_benchmark_suite(sycl::queue& q) {
  constexpr double flops = 2.0 * K * M * M;
  
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "M=" << M << ", K=" << K << ", stride=" << stride 
            << ", local_size=" << local_size << ", tile_size=" << max_tile_size << "\n";
  std::cout << "Elements=" << (M*M) << ", Tiles=" << ((M*M)/(max_tile_size*max_tile_size))
            << ", Threads/tile=" << (stride/((M*M)/(max_tile_size*max_tile_size))) << "\n";
  std::cout << "========================================\n";

  auto* A = sycl::malloc_shared<Scalar>(K * M, q);
  auto* B = sycl::malloc_shared<Scalar>(K * M, q);
  auto* C = sycl::malloc_shared<Scalar>(M * M, q);

  const auto zero_C = [&]() {
    q.fill(C, Scalar{0}, M * M).wait();
  };

  // Initialize with thread-safe RNG
  #pragma omp parallel
  {
    #ifdef _OPENMP
    std::minstd_rand0 rng(123 + omp_get_thread_num());
    #else
    std::minstd_rand0 rng(123);
    #endif
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    #pragma omp for
    for (int i = 0; i < K * M; ++i) {
      A[i] = dist(rng);
      B[i] = dist(rng);
    }
  }

  q.wait();
  auto C_correct = compute_naive(K, M, A, B);

  std::cout << "\n=== Correctness Check ===\n";
  auto check_correctness = [&](const char* name, auto&& kernel) {
    zero_C();
    kernel().wait();
    bool correct = is_correct(C_correct, C, M);
    std::cout << std::setw(12) << name << ": " << std::boolalpha << correct << "\n";
    if (!correct) {
      std::cout << "  (skipping performance test due to incorrect results)\n";
    }
    return correct;
  };

  bool v4_correct = check_correctness("Variant 4", [&]() { return tsmttsm4<Scalar, M, stride, max_tile_size>(q, K, A, B, C); });
  bool v6_correct = check_correctness("Variant 6", [&]() { return tsmttsm6<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); });
  bool v9_correct = check_correctness("Variant 9", [&]() { return tsmttsm9<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); });

  std::cout << "\n=== Performance (warmup=" << warmup << ", runs=" << runs << ") ===\n";

  auto run_and_profile = [&](auto&& kernel) -> BenchmarkStats {
    for (int i = 0; i < warmup; i++) {
      zero_C();
      kernel().wait();
    }
    q.wait_and_throw();

    std::vector<double> times_ns(runs);

    for (int i = 0; i < runs; i++) {
      zero_C();
      auto evt = kernel();
      evt.wait();

      uint64_t start = evt.template get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t end = evt.template get_profiling_info<sycl::info::event_profiling::command_end>();
      times_ns[i] = static_cast<double>(end - start);
    }

    double sum = std::accumulate(times_ns.begin(), times_ns.end(), 0.0);
    double avg = sum / runs;
    double min_t = *std::min_element(times_ns.begin(), times_ns.end());
    double max_t = *std::max_element(times_ns.begin(), times_ns.end());

    double sq_sum = std::inner_product(times_ns.begin(), times_ns.end(), times_ns.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / runs - avg * avg);

    double gflops = flops / (avg * 1e-9) / 1e9;

    return {avg, min_t, max_t, stddev, gflops};
  };

  auto print_stats = [](const char* name, const BenchmarkStats& s) {
    std::cout << std::setw(12) << name << ": "
              << std::fixed << std::setprecision(4)
              << s.avg_ns / 1e6 << " ms (±" << s.stddev_ns / 1e6 << " ms), "
              << std::setprecision(2) << s.gflops << " GFLOP/s\n";
  };

  if (v4_correct) print_stats("Variant 4", run_and_profile([&]() { return tsmttsm4<Scalar, M, stride, max_tile_size>(q, K, A, B, C); }));
  if (v6_correct) print_stats("Variant 6", run_and_profile([&]() { return tsmttsm6<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); }));
  if (v9_correct) print_stats("Variant 9", run_and_profile([&]() { return tsmttsm9<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); }));

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

int main() {
  sycl::queue q{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::vendor>() << " "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  using Scalar = double;
  constexpr int K = 1024 * 1024;
  constexpr int warmup = 50;
  constexpr int runs = 100;

  // Test different M values with various stride and tile size settings
  // Format: M, stride, local_size, tile_size
  
  // M=4: 16 elements total
  run_benchmark_suite<Scalar, 4, K, 4096, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 4, K, 4096, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 4, K, 8192, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 4, K, 8192, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 4, K, 16384, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 4, K, 16384, 32, 4, warmup, runs>(q);
  
  // M=8: 64 elements total
  run_benchmark_suite<Scalar, 8, K, 4096, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 8, K, 4096, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 8, K, 8192, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 8, K, 8192, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 8, K, 16384, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 8, K, 16384, 32, 4, warmup, runs>(q);
  
  // M=16: 256 elements total
  run_benchmark_suite<Scalar, 16, K, 8192, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 16, K, 8192, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 16, K, 16384, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 16, K, 16384, 32, 4, warmup, runs>(q);
  run_benchmark_suite<Scalar, 16, K, 32768, 32, 2, warmup, runs>(q);
  run_benchmark_suite<Scalar, 16, K, 32768, 32, 4, warmup, runs>(q);

  return 0;
}
