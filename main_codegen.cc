#include <sycl/sycl.hpp>
#include "tsmttsm_kernels.hh"

int main() {
  sycl::queue q{{sycl::property::queue::in_order{}, 
                 sycl::property::queue::enable_profiling{}}};

  // Run benchmarks for double type
  std::cout << "=== Benchmarking with double ===\n";
  run_all_benchmarks<double>(q);

  return 0;
}
