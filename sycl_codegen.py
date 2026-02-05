#!/usr/bin/env python3
"""
SYCL TSMTTSM Kernel Code Generator

Generates optimized SYCL kernels for tall-skinny matrix transpose multiply (A^T * B = C)
with various optimization strategies.

Usage:
    python3 sycl_codegen.py > generated_kernels.hh
"""

import sys
import argparse


def generate_kernel(M, N, TM, TN, stride, local_size=32, 
                   transposed=True, leap_frog=False, 
                   reduction="global", unroll=1, dtype="float"):
    """
    Generate SYCL kernel code for TSMTTSM operation.
    
    Args:
        M, N: Matrix dimensions (M x M output)
        TM, TN: Tile dimensions
        stride: Global thread count
        local_size: Workgroup size for two-level reduction
        transposed: Use transposed tiling pattern for coalescing
        leap_frog: Use double buffering (prefetching)
        reduction: "global" (global atomics only) or "twolevel" (workgroup + global)
        unroll: Loop unrolling factor for K iteration loop
        dtype: Data type (float, double)
    """
    
    mthreads = (M + TM - 1) // TM
    nthreads = (N + TN - 1) // TN
    threads_per_tile = mthreads * nthreads
    num_tiles = threads_per_tile
    
    kernel_name = f"tsmttsm_gen_{M}_{N}_{TM}_{TN}"
    if leap_frog:
        kernel_name += "_lf"
    if reduction == "twolevel":
        kernel_name += "_2l"
    if transposed:
        kernel_name += "_tr"
    
    code = []
    
    # Function signature
    code.append(f"/* Generated kernel: M={M}, N={N}, TM={TM}, TN={TN}, stride={stride}")
    code.append(f"   transposed={transposed}, leap_frog={leap_frog}, reduction={reduction} */")
    code.append(f"template <typename T>")
    
    if reduction == "twolevel":
        code.append(f"sycl::event {kernel_name}(sycl::queue& q, int K, T* A, T* B, T* C) {{")
        code.append(f"  constexpr int TM = {TM};")
        code.append(f"  constexpr int TN = {TN};")
        code.append(f"  constexpr int local_size = {local_size};")
        code.append(f"  constexpr int stride = {stride};")
        code.append(f"  return q.submit([&](sycl::handler& cgh) {{")
        code.append(f"    sycl::local_accessor<T, 2> C_shared({{TM, TN}}, cgh);")
        code.append(f"    cgh.parallel_for(sycl::nd_range<1>{{stride, local_size}}, [=](sycl::nd_item<1> item) {{")
    else:
        code.append(f"sycl::event {kernel_name}(sycl::queue& q, int K, T* A, T* B, T* C) {{")
        code.append(f"  constexpr int stride = {stride};")
        code.append(f"  return q.submit([&](sycl::handler& cgh) {{")
        code.append(f"    cgh.parallel_for(stride, [=](auto item) {{")
    
    # Thread indexing
    if reduction == "twolevel":
        code.append(f"      int tid = item.get_global_linear_id();")
        code.append(f"      int lid = item.get_local_linear_id();")
        code.append(f"      int group_id = item.get_group(0);")
        code.append(f"      int num_groups = stride / local_size;")
        code.append(f"")
        code.append(f"      int tile_idx = group_id % {num_tiles};")
        code.append(f"      int midx = tile_idx % {mthreads};")
        code.append(f"      int nidx = tile_idx / {mthreads};")
        code.append(f"")
        code.append(f"      int k_start = (group_id / {num_tiles}) * local_size + lid;")
        code.append(f"      int k_stride = (num_groups / {num_tiles}) * local_size;")
    else:
        code.append(f"      int tid = item[0];")
        code.append(f"      int tile_idx = tid % {num_tiles};")
        code.append(f"      int midx = tile_idx % {mthreads};")
        code.append(f"      int nidx = tile_idx / {mthreads};")
        code.append(f"")
        code.append(f"      int k_start = tid / {num_tiles};")
        code.append(f"      int k_stride = stride / {num_tiles};")
    
    code.append(f"")
    
    # Initialize shared memory for two-level reduction
    if reduction == "twolevel":
        code.append(f"      constexpr int TM = {TM};")
        code.append(f"      constexpr int TN = {TN};")
        code.append(f"      if (lid == 0) {{")
        code.append(f"        for (int m = 0; m < TM; ++m)")
        code.append(f"          for (int n = 0; n < TN; ++n)")
        code.append(f"            C_shared[m][n] = T(0);")
        code.append(f"      }}")
        code.append(f"      item.barrier(sycl::access::fence_space::local_space);")
        code.append(f"")
    
    # Declare accumulator variables (individual scalars, not arrays!)
    code.append(f"      // Accumulator variables (one per tile element)")
    for m in range(TM):
        decls = []
        for n in range(TN):
            decls.append(f"tS{m}_{n} = T(0)")
        code.append(f"      T {', '.join(decls)};")
    code.append(f"")
    
    # Helper function to generate load expression
    def get_load_expr(array, X, TX, x, xthreads, tidx_var, idx_var):
        """Generate memory load expression."""
        if transposed:
            # Transposed: A[idx*M + x*xthreads + tidx]
            return f"{array}[({idx_var}) * {X} + {x} * {xthreads} + {tidx_var}]"
        else:
            # Contiguous: A[idx*M + tidx*TX + x]
            return f"{array}[({idx_var}) * {X} + {tidx_var} * {TX} + {x}]"
    
    # Double buffering: declare buffer variables
    if leap_frog:
        code.append(f"      // Double buffering: declare now/next buffers")
        for u in range(unroll):
            for m in range(TM):
                code.append(f"      T vANow_{m}_{u} = T(0);")
            for n in range(TN):
                code.append(f"      T vBNow_{n}_{u} = T(0);")
        code.append(f"")
        
        # Prefetch first iteration
        code.append(f"      // Prefetch first iteration")
        code.append(f"      if (k_start < K) {{")
        for u in range(unroll):
            for m in range(TM):
                expr = get_load_expr("A", M, TM, m, mthreads, "midx", f"k_start + {u}*k_stride")
                code.append(f"        vANow_{m}_{u} = {expr};")
            for n in range(TN):
                expr = get_load_expr("B", N, TN, n, nthreads, "nidx", f"k_start + {u}*k_stride")
                code.append(f"        vBNow_{n}_{u} = {expr};")
        code.append(f"      }}")
        code.append(f"")
    
    # Main loop
    code.append(f"      // Main computation loop")
    if leap_frog:
        code.append(f"      int k;")
    loop_bound = f"K - k_stride*{unroll}" if leap_frog else "K"
    k_decl = "k = k_start" if leap_frog else "int k = k_start"
    code.append(f"      for ({k_decl}; k < {loop_bound}; k += k_stride*{unroll}) {{")
    
    # Process unroll iterations
    for u in range(unroll):
        if leap_frog:
            # Prefetch next iteration (unconditionally - we know it's safe)
            code.append(f"        // Prefetch next iteration (u={u})")
            for m in range(TM):
                expr = get_load_expr("A", M, TM, m, mthreads, "midx", f"k + k_stride*({unroll} + {u})")
                code.append(f"        T vANext_{m}_{u} = {expr};")
            for n in range(TN):
                expr = get_load_expr("B", N, TN, n, nthreads, "nidx", f"k + k_stride*({unroll} + {u})")
                code.append(f"        T vBNext_{n}_{u} = {expr};")
            code.append(f"")
            
            # Compute with current values
            code.append(f"        // Compute with current buffers (u={u})")
            for m in range(TM):
                for n in range(TN):
                    code.append(f"        tS{m}_{n} += vANow_{m}_{u} * vBNow_{n}_{u};")
            code.append(f"")
            
            # Swap buffers
            code.append(f"        // Swap buffers (u={u})")
            for m in range(TM):
                code.append(f"        vANow_{m}_{u} = vANext_{m}_{u};")
            for n in range(TN):
                code.append(f"        vBNow_{n}_{u} = vBNext_{n}_{u};")
            if u < unroll - 1:
                code.append(f"")
        else:
            # Load current iteration
            code.append(f"        // Load values (u={u})")
            for m in range(TM):
                expr = get_load_expr("A", M, TM, m, mthreads, "midx", f"k + {u}*k_stride")
                code.append(f"        T vA_{m}_{u} = {expr};")
            for n in range(TN):
                expr = get_load_expr("B", N, TN, n, nthreads, "nidx", f"k + {u}*k_stride")
                code.append(f"        T vB_{n}_{u} = {expr};")
            code.append(f"")
            
            # Compute
            code.append(f"        // Compute outer product (u={u})")
            for m in range(TM):
                for n in range(TN):
                    code.append(f"        tS{m}_{n} += vA_{m}_{u} * vB_{n}_{u};")
            if u < unroll - 1:
                code.append(f"")
    
    code.append(f"      }}")
    code.append(f"")
    
    # Handle last iteration for leap frog - just like CUDA version
    if leap_frog:
        code.append(f"      // Process final iteration (data in vANow/vBNow)")
        code.append(f"      if (k < K) {{")
        for u in range(unroll):
            for m in range(TM):
                for n in range(TN):
                    code.append(f"        tS{m}_{n} += vANow_{m}_{u} * vBNow_{n}_{u};")
        code.append(f"      }}")
        code.append(f"")
    
    # Reduction
    if reduction == "twolevel":
        code.append(f"      // Two-level reduction: local then global")
        code.append(f"      item.barrier(sycl::access::fence_space::local_space);")
        code.append(f"")
        for m in range(TM):
            for n in range(TN):
                code.append(f"      {{")
                code.append(f"        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,")
                code.append(f"                         sycl::access::address_space::local_space> local_ref{m}_{n}(C_shared[{m}][{n}]);")
                code.append(f"        local_ref{m}_{n} += tS{m}_{n};")
                code.append(f"      }}")
        code.append(f"")
        code.append(f"      item.barrier(sycl::access::fence_space::local_space);")
        code.append(f"")
        code.append(f"      if (lid == 0) {{")
        for m in range(TM):
            for n in range(TN):
                if transposed:
                    addr = f"({m} * {mthreads} + midx) * {N} + ({n} * {nthreads} + nidx)"
                else:
                    addr = f"({m} + midx * {TM}) * {N} + ({n} + nidx * {TN})"
                code.append(f"        {{")
                code.append(f"          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,")
                code.append(f"                           sycl::access::address_space::global_space> global_ref{m}_{n}(C[{addr}]);")
                code.append(f"          global_ref{m}_{n} += C_shared[{m}][{n}];")
                code.append(f"        }}")
        code.append(f"      }}")
    else:
        code.append(f"      // Global atomic reduction")
        for m in range(TM):
            for n in range(TN):
                if transposed:
                    addr = f"({m} * {mthreads} + midx) * {N} + ({n} * {nthreads} + nidx)"
                else:
                    addr = f"({m} + midx * {TM}) * {N} + ({n} + nidx * {TN})"
                code.append(f"      {{")
                code.append(f"        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,")
                code.append(f"                         sycl::access::address_space::global_space> C_ref{m}_{n}(C[{addr}]);")
                code.append(f"        C_ref{m}_{n} += tS{m}_{n};")
                code.append(f"      }}")
    
    code.append(f"    }});")
    code.append(f"  }});")
    code.append(f"}}")
    code.append(f"")
    
    return "\n".join(code)


def generate_benchmark_code(M_list, tile_sizes, stride, local_size):
    """Generate benchmark function that tests all generated kernels."""
    code = []
    
    code.append("// ============================================")
    code.append("// Benchmark Infrastructure")
    code.append("// ============================================")
    code.append("")
    code.append("#include <iostream>")
    code.append("#include <iomanip>")
    code.append("#include <vector>")
    code.append("#include <cmath>")
    code.append("#include <numeric>")
    code.append("#include <algorithm>")
    code.append("#include <string>")
    code.append("")
    code.append("struct KernelBenchmarkResult {")
    code.append("  std::string name;")
    code.append("  bool correct;")
    code.append("  double avg_ms;")
    code.append("  double min_ms;")
    code.append("  double stddev_ms;")
    code.append("  double gflops;")
    code.append("};")
    code.append("")
    code.append("template <typename T>")
    code.append("std::vector<T> compute_reference(int K, int M, T* A, T* B) {")
    code.append("  std::vector<T> C(M * M, T(0));")
    code.append("  for (int m = 0; m < M; ++m) {")
    code.append("    for (int n = 0; n < M; ++n) {")
    code.append("      for (int k = 0; k < K; ++k) {")
    code.append("        C[m * M + n] += A[k * M + m] * B[k * M + n];")
    code.append("      }")
    code.append("    }")
    code.append("  }")
    code.append("  return C;")
    code.append("}")
    code.append("")
    code.append("template <typename T>")
    code.append("bool check_correctness(const std::vector<T>& C_ref, T* C, int M, T abs_tol = 1e-4, T rel_tol = 1e-8) {")
    code.append("  for (int i = 0; i < M * M; ++i) {")
    code.append("    T diff = std::abs(C_ref[i] - C[i]);")
    code.append("    T mag = std::max(std::abs(C_ref[i]), std::abs(C[i]));")
    code.append("    if (diff > abs_tol && diff > rel_tol * mag) {")
    code.append("      std::cerr << \"Mismatch at \" << i << \": expected \" << C_ref[i]")
    code.append("                << \", got \" << C[i] << \" (diff=\" << diff << \")\\n\";")
    code.append("      return false;")
    code.append("    }")
    code.append("  }")
    code.append("  return true;")
    code.append("}")
    code.append("")
    code.append("template <typename T>")
    code.append("void run_all_benchmarks(sycl::queue& q, int K = 1024*1024, int warmup = 50, int runs = 100) {")
    code.append("  std::cout << \"Device: \" << q.get_device().get_info<sycl::info::device::name>() << \"\\n\\n\";")
    code.append("")
    code.append("  std::vector<KernelBenchmarkResult> results;")
    code.append("")
    
    # Generate benchmark code for each M
    for M in M_list:
        code.append(f"  // Benchmarks for M={M}")
        code.append(f"  {{")
        code.append(f"    constexpr int M = {M};")
        code.append(f"    const double flops = 2.0 * K * M * M;")
        code.append(f"")
        code.append(f"    auto* A = sycl::malloc_shared<T>(K * M, q);")
        code.append(f"    auto* B = sycl::malloc_shared<T>(K * M, q);")
        code.append(f"    auto* C = sycl::malloc_shared<T>(M * M, q);")
        code.append(f"")
        code.append(f"    // Initialize data")
        code.append(f"    for (int i = 0; i < K * M; ++i) {{")
        code.append(f"      A[i] = T(rand()) / RAND_MAX - 0.5;")
        code.append(f"      B[i] = T(rand()) / RAND_MAX - 0.5;")
        code.append(f"    }}")
        code.append(f"    q.wait();")
        code.append(f"")
        code.append(f"    auto C_ref = compute_reference(K, M, A, B);")
        code.append(f"")
        code.append(f"    auto benchmark_kernel = [&](const std::string& name, auto&& kernel_func) {{")
        code.append(f"      q.fill(C, T(0), M * M).wait();")
        code.append(f"")
        code.append(f"      // Correctness check")
        code.append(f"      kernel_func().wait();")
        code.append(f"      bool correct = check_correctness(C_ref, C, M);")
        code.append(f"")
        code.append(f"      if (!correct) {{")
        code.append(f"        results.push_back({{name, false, 0, 0, 0, 0}});")
        code.append(f"        return;")
        code.append(f"      }}")
        code.append(f"")
        code.append(f"      // Warmup")
        code.append(f"      for (int i = 0; i < warmup; ++i) {{")
        code.append(f"        q.fill(C, T(0), M * M).wait();")
        code.append(f"        kernel_func().wait();")
        code.append(f"      }}")
        code.append(f"")
        code.append(f"      // Benchmark")
        code.append(f"      std::vector<double> times(runs);")
        code.append(f"      for (int i = 0; i < runs; ++i) {{")
        code.append(f"        q.fill(C, T(0), M * M).wait();")
        code.append(f"        auto evt = kernel_func();")
        code.append(f"        evt.wait();")
        code.append(f"        auto start = evt.template get_profiling_info<sycl::info::event_profiling::command_start>();")
        code.append(f"        auto end = evt.template get_profiling_info<sycl::info::event_profiling::command_end>();")
        code.append(f"        times[i] = (end - start) / 1e6; // convert to ms")
        code.append(f"      }}")
        code.append(f"")
        code.append(f"      double avg = std::accumulate(times.begin(), times.end(), 0.0) / runs;")
        code.append(f"      double min_t = *std::min_element(times.begin(), times.end());")
        code.append(f"      double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);")
        code.append(f"      double stddev = std::sqrt(sq_sum / runs - avg * avg);")
        code.append(f"      double gflops = flops / (avg * 1e-3) / 1e9;")
        code.append(f"")
        code.append(f"      results.push_back({{name, true, avg, min_t, stddev, gflops}});")
        code.append(f"    }};")
        code.append(f"")
        
        # Generate benchmark calls for each variant
        for TM in tile_sizes:
            if M % TM != 0:
                continue
            
            base_name = f"M{M}_T{TM}"
            
            # Variant 1: Basic
            kernel_name = f"tsmttsm_gen_{M}_{M}_{TM}_{TM}_tr"
            code.append(f"    benchmark_kernel(\"{base_name}_basic\", [&]() {{ return {kernel_name}(q, K, A, B, C); }});")
            
            # Variant 2: Leap frog
            kernel_name = f"tsmttsm_gen_{M}_{M}_{TM}_{TM}_lf_tr"
            code.append(f"    benchmark_kernel(\"{base_name}_leapfrog\", [&]() {{ return {kernel_name}(q, K, A, B, C); }});")
            
            # Variant 3: Two-level
            kernel_name = f"tsmttsm_gen_{M}_{M}_{TM}_{TM}_2l_tr"
            code.append(f"    benchmark_kernel(\"{base_name}_2level\", [&]() {{ return {kernel_name}(q, K, A, B, C); }});")
            
            # Variant 4: All opts
            kernel_name = f"tsmttsm_gen_{M}_{M}_{TM}_{TM}_lf_2l_tr"
            code.append(f"    benchmark_kernel(\"{base_name}_all\", [&]() {{ return {kernel_name}(q, K, A, B, C); }});")
            code.append(f"")
        
        code.append(f"    sycl::free(A, q);")
        code.append(f"    sycl::free(B, q);")
        code.append(f"    sycl::free(C, q);")
        code.append(f"  }}")
        code.append(f"")
    
    # Print results
    code.append("  // Print results")
    code.append("  std::cout << \"\\n\" << std::string(100, '=') << \"\\n\";")
    code.append("  std::cout << \"BENCHMARK RESULTS\\n\";")
    code.append("  std::cout << std::string(100, '=') << \"\\n\\n\";")
    code.append("  std::cout << std::setw(25) << \"Kernel\" ")
    code.append("            << std::setw(10) << \"Correct\" ")
    code.append("            << std::setw(12) << \"Avg (ms)\"")
    code.append("            << std::setw(12) << \"Min (ms)\"")
    code.append("            << std::setw(12) << \"Stddev\"")
    code.append("            << std::setw(12) << \"GFLOP/s\" << \"\\n\";")
    code.append("  std::cout << std::string(100, '-') << \"\\n\";")
    code.append("  ")
    code.append("  for (const auto& r : results) {")
    code.append("    std::cout << std::setw(25) << r.name")
    code.append("              << std::setw(10) << (r.correct ? \"PASS\" : \"FAIL\")")
    code.append("              << std::fixed << std::setprecision(4);")
    code.append("    if (r.correct) {")
    code.append("      std::cout << std::setw(12) << r.avg_ms")
    code.append("                << std::setw(12) << r.min_ms")
    code.append("                << std::setw(12) << r.stddev_ms")
    code.append("                << std::setprecision(2)")
    code.append("                << std::setw(12) << r.gflops;")
    code.append("    }")
    code.append("    std::cout << \"\\n\";")
    code.append("  }")
    code.append("  std::cout << std::string(100, '=') << \"\\n\";")
    code.append("}")
    code.append("")
    
    return "\n".join(code)


def main():
    parser = argparse.ArgumentParser(description='Generate SYCL TSMTTSM kernels')
    parser.add_argument('--M', type=int, nargs='+', default=[4, 8, 16],
                       help='Matrix sizes to generate (default: 4 8 16)')
    parser.add_argument('--tile-sizes', type=int, nargs='+', default=[2, 4],
                       help='Tile sizes to generate (default: 2 4)')
    parser.add_argument('--stride', type=int, default=16384,
                       help='Number of threads (default: 16384)')
    parser.add_argument('--local-size', type=int, default=32,
                       help='Workgroup size (default: 32)')
    parser.add_argument('--unroll', type=int, default=1,
                       help='K-loop unrolling factor (default: 1)')
    parser.add_argument('--with-benchmark', action='store_true',
                       help='Generate benchmark code')
    args = parser.parse_args()
    
    print("#pragma once")
    print("")
    print("#include <sycl/sycl.hpp>")
    print("")
    print("/* Auto-generated SYCL kernels for TSMTTSM operation")
    print(" * Generated with optimizations:")
    print(" *   - Individual scalar accumulators (not arrays)")
    print(" *   - Fully unrolled compute loops")
    print(" *   - Transposed tiling for coalesced memory access")
    print(" *   - Optional double buffering (leap-frog prefetching)")
    print(" *   - Optional two-level reduction (workgroup + global atomics)")
    print(f" *   - K-loop unrolling factor: {args.unroll}")
    print(" */")
    print("")
    
    # Generate variants
    for M in args.M:
        print(f"// ============================================")
        print(f"// Kernels for M = {M}")
        print(f"// ============================================")
        print("")
        
        for TM in args.tile_sizes:
            if M % TM != 0:
                continue
                
            # Variant 1: Basic with transposed tiling
            print(generate_kernel(M, M, TM, TM, args.stride, args.local_size,
                                transposed=True, leap_frog=False, reduction="global", unroll=args.unroll))
            
            # Variant 2: With double buffering
            print(generate_kernel(M, M, TM, TM, args.stride, args.local_size,
                                transposed=True, leap_frog=True, reduction="global", unroll=args.unroll))
            
            # Variant 3: Two-level reduction
            print(generate_kernel(M, M, TM, TM, args.stride, args.local_size,
                                transposed=True, leap_frog=False, reduction="twolevel", unroll=args.unroll))
            
            # Variant 4: All optimizations
            print(generate_kernel(M, M, TM, TM, args.stride, args.local_size,
                                transposed=True, leap_frog=True, reduction="twolevel", unroll=args.unroll))
    
    if args.with_benchmark:
        print(generate_benchmark_code(args.M, args.tile_sizes, args.stride, args.local_size))
    
    print("// End of generated code")


if __name__ == "__main__":
    main()
