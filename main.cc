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
bool is_correct(const std::vector<T>& C_correct, T* C, T rel_tol = 1e-6, T abs_tol = 1e-10) {
  for (std::size_t i = 0; i < C_correct.size(); ++i) {
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

int main() {
  sycl::queue q{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::vendor>() << " "
            << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

  constexpr int K = 1024 * 1024;
  constexpr int M = 16;
  constexpr int stride = 32768 / 2; // aka global_size in nd_range kernels
  constexpr int local_size = 32;  
  constexpr int max_tile_size = 4;
  constexpr int warmup = 100;
  constexpr int runs = 200;

  const auto flops = 2.0 * K * M * M;

  using Scalar = float;

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

  std::cout << "=== Correctness Check (stride=" << stride << ") ===\n";
  auto check_correctness = [&](const char* name, auto&& kernel) {
    zero_C();
    kernel().wait();
    std::cout << std::setw(12) << name << ": " << std::boolalpha
              << is_correct(C_correct, C) << "\n";
  };

  check_correctness("Variant 1", [&]() { return tsmttsm<Scalar, M, stride>(q, K, A, B, C); });
  check_correctness("Variant 2", [&]() { return tsmttsm2<Scalar, M, stride, max_tile_size>(q, K, A, B, C); });
  check_correctness("Variant 3", [&]() { return tsmttsm3<Scalar, M, stride, max_tile_size>(q, K, A, B, C); });
  check_correctness("Variant 4", [&]() { return tsmttsm4<Scalar, M, stride, max_tile_size>(q, K, A, B, C); });
  check_correctness("Variant 5", [&]() { return tsmttsm5<Scalar, M, stride, max_tile_size>(q, K, A, B, C); });
  check_correctness("Variant 6", [&]() { return tsmttsm6<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); });

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
              << s.avg_ns / 1e6 << " ms (±" << s.stddev_ns / 1e6 << "), "
              << "min=" << s.min_ns / 1e6 << ", max=" << s.max_ns / 1e6 << " ms, "
              << std::setprecision(2) << s.gflops << " GFLOP/s\n";
  };

  print_stats("Variant 1", run_and_profile([&]() { return tsmttsm<Scalar, M, stride>(q, K, A, B, C); }));
  print_stats("Variant 2", run_and_profile([&]() { return tsmttsm2<Scalar, M, stride, max_tile_size>(q, K, A, B, C); }));
  print_stats("Variant 3", run_and_profile([&]() { return tsmttsm3<Scalar, M, stride, max_tile_size>(q, K, A, B, C); }));
  print_stats("Variant 4", run_and_profile([&]() { return tsmttsm4<Scalar, M, stride, max_tile_size>(q, K, A, B, C); }));
  print_stats("Variant 5", run_and_profile([&]() { return tsmttsm5<Scalar, M, stride, max_tile_size>(q, K, A, B, C); }));
  print_stats("Variant 6", run_and_profile([&]() { return tsmttsm6<Scalar, M, stride, local_size, max_tile_size>(q, K, A, B, C); }));

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  return 0;
}
