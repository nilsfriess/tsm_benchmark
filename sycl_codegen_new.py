#!/usr/bin/env python3
"""
SYCL TSMTTSM Kernel Code Generator v2

Generates optimized SYCL kernels for A^T * B = C (tall-skinny matrix multiply).

Design principles vs v1:
  - Two-level reduction removed (compiler-hostile via local_accessor, unreliable)
  - Sub-group reduction replaces it: standard SYCL 2020 reduce_over_group,
    no shared memory, portable across NVIDIA/AMD/Intel.
    Sub-group size is queried at runtime, no compiler attributes needed.
  - Transposed tiling always on (provably better for coalescing)
  - Leap-frog is a separate independent axis from sub-group reduction

Variants generated per (M, TM):
  basic    : global range, scalar accumulators, global atomics
  lf       : basic + double-buffered prefetch
  sg       : nd_range, sub_group reduce_over_group, then single leader atomic
  sg_lf    : sg + double-buffered prefetch
"""

import sys
import argparse
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lcm(a, b):
    return a * b // math.gcd(a, b)


def _accum_decls(TM, TN, indent="      "):
    """Declare TM*TN scalar accumulator variables initialised to T(0)."""
    lines = []
    for m in range(TM):
        parts = ", ".join(f"tS{m}_{n} = T(0)" for n in range(TN))
        lines.append(f"{indent}T {parts};")
    return lines


def _compute_block(TM, TN, a_vars, b_vars, indent="        "):
    """Generate outer-product accumulate: tS[m][n] += a[m] * b[n]."""
    lines = []
    for m in range(TM):
        for n in range(TN):
            lines.append(f"{indent}tS{m}_{n} += {a_vars[m]} * {b_vars[n]};")
    return lines


# ---------------------------------------------------------------------------
# Transposed load expressions
# ---------------------------------------------------------------------------
# Memory layout: A is stored as A[K][M], i.e. row-major with row = k-index.
# In the transposed tiling scheme each thread handles a horizontal slice of
# one tile column: consecutive threads in the midx dimension load consecutive
# elements → coalesced.
#   A[(k) * M + m * mthreads + midx]
#   B[(k) * N + n * nthreads + nidx]

def _la(M, mthreads, m, k_expr):
    return f"A[({k_expr}) * {M} + {m} * {mthreads} + midx]"

def _lb(N, nthreads, n, k_expr):
    return f"B[({k_expr}) * {N} + {n} * {nthreads} + nidx]"


# ---------------------------------------------------------------------------
# Leap-frog body generator (shared between basic and sg variants)
# ---------------------------------------------------------------------------

def _leapfrog_body(TM, TN, M, N, mthreads, nthreads, unroll, k_var="k_start"):
    """
    Return (declarations, initial_prefetch, main_loop, drain) as lists of lines.
    k_var is the name of the k_start variable for this thread/subgroup lane.
    The caller must emit:
        declarations
        initial_prefetch
        main_loop  (contains its own 'int k;' + for-loop)
        drain
    """
    decls = []
    for u in range(unroll):
        for m in range(TM): decls.append(f"      T vA_{m}_{u} = T(0);")
        for n in range(TN): decls.append(f"      T vB_{n}_{u} = T(0);")
    decls.append("")

    # Initial prefetch: one if-guard per unroll slot
    prefetch = ["      // Initial prefetch"]
    for u in range(unroll):
        cond = f"{k_var} < K" if u == 0 else f"{k_var} + {u} * k_stride < K"
        prefetch.append(f"      if ({cond}) {{")
        for m in range(TM):
            prefetch.append(f"        vA_{m}_{u} = {_la(M, mthreads, m, f'{k_var} + {u} * k_stride')};")
        for n in range(TN):
            prefetch.append(f"        vB_{n}_{u} = {_lb(N, nthreads, n, f'{k_var} + {u} * k_stride')};")
        prefetch.append(f"      }}")
    prefetch.append("")

    # Main loop
    loop = [
        f"      int k;",
        f"      for (k = {k_var}; k < K - k_stride * {unroll}; k += k_stride * {unroll}) {{",
    ]
    for u in range(unroll):
        next_k = f"k + k_stride * ({unroll} + {u})"
        for m in range(TM):
            expr = _la(M, mthreads, m, next_k)
            loop.append(f"        const T nA_{m}_{u} = ({next_k} < K) ? {expr} : T(0);")
        for n in range(TN):
            expr = _lb(N, nthreads, n, next_k)
            loop.append(f"        const T nB_{n}_{u} = ({next_k} < K) ? {expr} : T(0);")
        loop.extend(_compute_block(TM, TN,
                                   [f"vA_{m}_{u}" for m in range(TM)],
                                   [f"vB_{n}_{u}" for n in range(TN)]))
        for m in range(TM): loop.append(f"        vA_{m}_{u} = nA_{m}_{u};")
        for n in range(TN): loop.append(f"        vB_{n}_{u} = nB_{n}_{u};")
    loop.append("      }")
    loop.append("")

    # Drain remaining (per-u guard because last k_start may differ per slot)
    drain = ["      // Drain remaining"]
    for u in range(unroll):
        cond = "k < K" if u == 0 else f"k + {u} * k_stride < K"
        drain.append(f"      if ({cond}) {{")
        drain.extend(_compute_block(TM, TN,
                                    [f"vA_{m}_{u}" for m in range(TM)],
                                    [f"vB_{n}_{u}" for n in range(TN)],
                                    indent="        "))
        drain.append("      }")
    drain.append("")

    return decls, prefetch, loop, drain


# ---------------------------------------------------------------------------
# Kernel generators
# ---------------------------------------------------------------------------

def generate_basic_kernel(M, N, TM, TN, stride, leap_frog=False, unroll=1):
    """
    Basic kernel: sycl::range (no nd_range), global atomics after K-loop.
    Transposed tiling always on.
    """
    assert M % TM == 0 and N % TN == 0
    mthreads = M // TM
    nthreads = N // TN
    num_tiles = mthreads * nthreads

    suffix = "_lf" if leap_frog else ""
    name = f"tsmttsm_{M}_{N}_{TM}_{TN}{suffix}"

    c = []
    c.append(f"/* {name}: M={M} N={N} TM={TM} TN={TN} "
             f"stride={stride} leap_frog={leap_frog} unroll={unroll} */")
    c.append("template <typename T>")
    c.append(f"sycl::event {name}(sycl::queue& q, int K, const T* A, const T* B, T* C) {{")
    c.append(f"  constexpr int stride = {stride};")
    c.append(f"  return q.submit([&](sycl::handler& cgh) {{")
    c.append(f"    cgh.parallel_for(sycl::range<1>{{stride}}, [=](sycl::item<1> item) {{")
    c.append(f"      const int tid    = item[0];")
    c.append(f"      const int midx   = (tid % {num_tiles}) % {mthreads};")
    c.append(f"      const int nidx   = (tid % {num_tiles}) / {mthreads};")
    c.append(f"      const int k_start  = tid / {num_tiles};")
    c.append(f"      const int k_stride = stride / {num_tiles};")
    c.append("")
    c.extend(_accum_decls(TM, TN))
    c.append("")

    if not leap_frog:
        if unroll == 1:
            c.append(f"      for (int k = k_start; k < K; k += k_stride) {{")
            for m in range(TM):
                c.append(f"        const T a{m} = {_la(M, mthreads, m, 'k')};")
            for n in range(TN):
                c.append(f"        const T b{n} = {_lb(N, nthreads, n, 'k')};")
            c.extend(_compute_block(TM, TN,
                                    [f"a{m}" for m in range(TM)],
                                    [f"b{n}" for n in range(TN)]))
            c.append("      }")
        else:
            # Fast path: loop while all unroll slots are in-bounds; scalar tail
            c.append(f"      {{")
            c.append(f"        int k = k_start;")
            c.append(f"        for (; k + {unroll - 1} * k_stride < K; k += k_stride * {unroll}) {{")
            for u in range(unroll):
                for m in range(TM):
                    c.append(f"          const T a{m}_{u} = {_la(M, mthreads, m, f'k + {u} * k_stride')};")
                for n in range(TN):
                    c.append(f"          const T b{n}_{u} = {_lb(N, nthreads, n, f'k + {u} * k_stride')};")
                c.extend(_compute_block(TM, TN,
                                        [f"a{m}_{u}" for m in range(TM)],
                                        [f"b{n}_{u}" for n in range(TN)],
                                        indent="          "))
            c.append("        }")
            c.append("        // Scalar tail for remaining k values")
            c.append("        for (; k < K; k += k_stride) {")
            for m in range(TM):
                c.append(f"          const T a{m} = {_la(M, mthreads, m, 'k')};")
            for n in range(TN):
                c.append(f"          const T b{n} = {_lb(N, nthreads, n, 'k')};")
            c.extend(_compute_block(TM, TN,
                                    [f"a{m}" for m in range(TM)],
                                    [f"b{n}" for n in range(TN)],
                                    indent="          "))
            c.append("        }")
            c.append("      }")
    else:
        decls, prefetch, loop, drain = _leapfrog_body(
            TM, TN, M, N, mthreads, nthreads, unroll, k_var="k_start")
        c.extend(decls)
        c.extend(prefetch)
        c.extend(loop)
        c.extend(drain)

    # Global atomic reduction
    c.append("      // Global atomic reduction")
    for m in range(TM):
        for n in range(TN):
            addr = f"({m} * {mthreads} + midx) * {N} + ({n} * {nthreads} + nidx)"
            c.append(f"      sycl::atomic_ref<T, sycl::memory_order::relaxed,")
            c.append(f"                       sycl::memory_scope::device,")
            c.append(f"          sycl::access::address_space::global_space>(C[{addr}]) += tS{m}_{n};")
    c.append("    });")
    c.append("  });")
    c.append("}")
    c.append("")
    return "\n".join(c)


def generate_sg_kernel(M, N, TM, TN, stride, local_size, sg_size,
                       leap_frog=False, unroll=1):
    """
    Sub-group kernel: nd_range, all threads in a sub_group cooperate on the
    same (midx, nidx) tile.  After the K-loop each thread holds a partial sum;
    sycl::reduce_over_group (standard SYCL 2020) reduces within the sub_group
    and the leader issues a single global atomic per output element.

    The actual sub-group size is queried at runtime via sub_group methods;
    no compiler-specific attributes are used.

    This is ~sg_size times fewer global atomics than the basic kernel.
    """
    assert M % TM == 0 and N % TN == 0
    mthreads = M // TM
    nthreads = N // TN
    num_tiles = mthreads * nthreads
    assert stride % sg_size == 0, "stride must be divisible by sg_size"
    total_sg = stride // sg_size
    assert total_sg % num_tiles == 0, \
        f"total_sg ({total_sg}) must be divisible by num_tiles ({num_tiles})"
    num_sg_per_wg = local_size // sg_size

    suffix = "_sg_lf" if leap_frog else "_sg"
    name = f"tsmttsm_{M}_{N}_{TM}_{TN}{suffix}"

    c = []
    c.append(f"/* {name}: M={M} N={N} TM={TM} TN={TN} stride={stride} "
             f"local_size={local_size} sg_size={sg_size} "
             f"leap_frog={leap_frog} unroll={unroll} */")
    c.append("template <typename T>")
    c.append(f"sycl::event {name}(sycl::queue& q, int K, const T* A, const T* B, T* C) {{")
    c.append(f"  constexpr int stride       = {stride};")
    c.append(f"  constexpr int local_size   = {local_size};")
    c.append(f"  constexpr int num_tiles    = {num_tiles};")
    c.append(f"  return q.submit([&](sycl::handler& cgh) {{")
    c.append(f"    cgh.parallel_for(")
    c.append(f"      sycl::nd_range<1>{{stride, local_size}},")
    c.append(f"      [=](sycl::nd_item<1> item) {{")

    # Sub-group indexing — use runtime queries so no compiler attribute is needed.
    # sg.get_local_linear_range()  = actual sub-group size chosen by the runtime
    # sg.get_group_linear_range()  = number of sub-groups per workgroup
    c.append(f"        auto sg                = item.get_sub_group();")
    c.append(f"        const int sg_lid       = (int)sg.get_local_linear_id();")
    c.append(f"        const int sg_size_rt   = (int)sg.get_local_linear_range();")
    c.append(f"        const int sg_per_wg    = (int)sg.get_group_linear_range();")
    c.append(f"        const int sg_gid       = (int)item.get_group(0) * sg_per_wg")
    c.append(f"                               + (int)sg.get_group_linear_id();")
    c.append(f"        const int total_sg_rt  = stride / sg_size_rt;")
    c.append("")
    c.append(f"        const int midx         = (sg_gid % num_tiles) % {mthreads};")
    c.append(f"        const int nidx         = (sg_gid % num_tiles) / {mthreads};")
    c.append(f"        const int k_start      = (sg_gid / num_tiles) * sg_size_rt + sg_lid;")
    c.append(f"        const int k_stride     = (total_sg_rt / num_tiles) * sg_size_rt;")
    c.append("")
    # Re-emit accum decls with deeper indent
    for m in range(TM):
        parts = ", ".join(f"tS{m}_{n} = T(0)" for n in range(TN))
        c.append(f"        T {parts};")
    c.append("")

    # Re-define load helpers at this indent level (they use midx/nidx/k_start/k_stride)
    def la(m, k_expr): return _la(M, mthreads, m, k_expr)
    def lb(n, k_expr): return _lb(N, nthreads, n, k_expr)

    def compute8(a_vars, b_vars):
        return _compute_block(TM, TN, a_vars, b_vars, indent="          ")

    if not leap_frog:
        if unroll == 1:
            c.append(f"        for (int k = k_start; k < K; k += k_stride) {{")
            for m in range(TM):
                c.append(f"          const T a{m} = {la(m, 'k')};")
            for n in range(TN):
                c.append(f"          const T b{n} = {lb(n, 'k')};")
            c.extend(compute8([f"a{m}" for m in range(TM)],
                               [f"b{n}" for n in range(TN)]))
            c.append("        }")
        else:
            c.append(f"        {{")
            c.append(f"          int k = k_start;")
            c.append(f"          for (; k + {unroll - 1} * k_stride < K; k += k_stride * {unroll}) {{")
            for u in range(unroll):
                for m in range(TM):
                    c.append(f"            const T a{m}_{u} = {la(m, f'k + {u} * k_stride')};")
                for n in range(TN):
                    c.append(f"            const T b{n}_{u} = {lb(n, f'k + {u} * k_stride')};")
                c.extend(_compute_block(TM, TN,
                                        [f"a{m}_{u}" for m in range(TM)],
                                        [f"b{n}_{u}" for n in range(TN)],
                                        indent="            "))
            c.append("          }")
            c.append("          for (; k < K; k += k_stride) {")
            for m in range(TM):
                c.append(f"            const T a{m} = {la(m, 'k')};")
            for n in range(TN):
                c.append(f"            const T b{n} = {lb(n, 'k')};")
            c.extend(_compute_block(TM, TN,
                                    [f"a{m}" for m in range(TM)],
                                    [f"b{n}" for n in range(TN)],
                                    indent="            "))
            c.append("          }")
            c.append("        }")
    else:
        # Leap-frog — reuse the body generator but adjust indentation manually
        decls, prefetch, loop, drain = _leapfrog_body(
            TM, TN, M, N, mthreads, nthreads, unroll, k_var="k_start")
        # The body generator uses "      " (6 spaces) indent; we need "        " (8)
        extra = "  "
        for line in decls + prefetch + loop + drain:
            c.append(extra + line)

    # Sub-group reduce then single leader atomic
    c.append("")
    c.append("        // Sub-group reduce then single leader global atomic")
    for m in range(TM):
        for n in range(TN):
            addr = f"({m} * {mthreads} + midx) * {N} + ({n} * {nthreads} + nidx)"
            c.append(f"        {{")
            c.append(f"          T r = sycl::reduce_over_group(sg, tS{m}_{n}, sycl::plus<T>());")
            c.append(f"          if (sg_lid == 0)")
            c.append(f"            sycl::atomic_ref<T, sycl::memory_order::relaxed,")
            c.append(f"                             sycl::memory_scope::device,")
            c.append(f"                sycl::access::address_space::global_space>(C[{addr}]) += r;")
            c.append(f"        }}")
    c.append("    });")
    c.append("  });")
    c.append("}")
    c.append("")
    return "\n".join(c)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def generate_benchmark(M_list, tile_sizes, stride, local_size, sg_size):
    """Generate run_all_benchmarks<T>() that exercises all four variants."""

    c = []
    c.append("// ============================================================")
    c.append("// Benchmark infrastructure")
    c.append("// ============================================================")
    c.append("")
    c.append("#include <iostream>")
    c.append("#include <iomanip>")
    c.append("#include <vector>")
    c.append("#include <string>")
    c.append("#include <cmath>")
    c.append("#include <numeric>")
    c.append("#include <algorithm>")
    c.append("")
    c.append("struct BenchResult {")
    c.append("  std::string name;")
    c.append("  bool correct;")
    c.append("  double avg_ms, min_ms, stddev_ms, gflops;")
    c.append("};")
    c.append("")
    c.append("template <typename T>")
    c.append("std::vector<T> reference_tsmttsm(int K, int M, const T* A, const T* B) {")
    c.append("  std::vector<T> C(M * M, T(0));")
    c.append("  for (int m = 0; m < M; ++m)")
    c.append("    for (int n = 0; n < M; ++n)")
    c.append("      for (int k = 0; k < K; ++k)")
    c.append("        C[m * M + n] += A[k * M + m] * B[k * M + n];")
    c.append("  return C;")
    c.append("}")
    c.append("")
    c.append("template <typename T>")
    c.append("bool check(const std::vector<T>& ref, const T* got, int M,")
    c.append("           T abs_tol = T(1e-4), T rel_tol = T(1e-6)) {")
    c.append("  for (int i = 0; i < M * M; ++i) {")
    c.append("    T diff = std::abs(ref[i] - got[i]);")
    c.append("    T mag  = std::max(std::abs(ref[i]), std::abs(got[i]));")
    c.append("    if (diff > abs_tol && diff > rel_tol * mag) {")
    c.append("      std::cerr << \"Mismatch at \" << i << \": ref=\" << ref[i]")
    c.append("                << \" got=\" << got[i] << \" diff=\" << diff << \"\\n\";")
    c.append("      return false;")
    c.append("    }")
    c.append("  }")
    c.append("  return true;")
    c.append("}")
    c.append("")
    c.append("template <typename T>")
    c.append("void run_all_benchmarks(sycl::queue& q, int K = 1 << 20,")
    c.append("                        int warmup = 10, int runs = 30) {")
    c.append("  std::cout << \"Device: \"")
    c.append("            << q.get_device().get_info<sycl::info::device::name>() << \"\\n\\n\";")
    c.append("  std::vector<BenchResult> results;")
    c.append("")

    for M in M_list:
        c.append(f"  // ----- M = {M} -----")
        c.append(f"  {{")
        c.append(f"    constexpr int M = {M};")
        c.append(f"    const double flops = 2.0 * K * (double)M * M;")
        c.append(f"    auto* Ah = sycl::malloc_host<T>(K * M, q);")
        c.append(f"    auto* Bh = sycl::malloc_host<T>(K * M, q);")
        c.append(f"    auto* Ch = sycl::malloc_host<T>(M * M, q);")
        c.append(f"    auto* A  = sycl::malloc_device<T>(K * M, q);")
        c.append(f"    auto* B  = sycl::malloc_device<T>(K * M, q);")
        c.append(f"    auto* C  = sycl::malloc_device<T>(M * M, q);")
        c.append(f"    for (int i = 0; i < K * M; ++i) {{")
        c.append(f"      Ah[i] = T(rand()) / RAND_MAX - T(0.5);")
        c.append(f"      Bh[i] = T(rand()) / RAND_MAX - T(0.5);")
        c.append(f"    }}")
        c.append(f"    q.memcpy(A, Ah, K * M * sizeof(T)).wait();")
        c.append(f"    q.memcpy(B, Bh, K * M * sizeof(T)).wait();")
        c.append(f"    // Reference on a small K for speed (correctness only)")
        c.append(f"    const int K_ref = std::min(K, 1024);")
        c.append(f"    auto Cref = reference_tsmttsm(K_ref, M, Ah, Bh);")
        c.append(f"")
        c.append(f"    auto bench = [&](const std::string& label, auto kernel_fn) {{")
        c.append(f"      // Correctness: run with K_ref on zeroed C")
        c.append(f"      q.fill(C, T(0), M * M).wait();")
        c.append(f"      kernel_fn(K_ref).wait();")
        c.append(f"      q.memcpy(Ch, C, M * M * sizeof(T)).wait();")
        c.append(f"      bool ok = check(Cref, Ch, M);")
        c.append(f"      if (!ok) {{ results.push_back({{label, false, 0,0,0,0}}); return; }}")
        c.append(f"")
        c.append(f"      // Warmup")
        c.append(f"      for (int i = 0; i < warmup; ++i) {{")
        c.append(f"        q.fill(C, T(0), M * M).wait();")
        c.append(f"        kernel_fn(K).wait();")
        c.append(f"      }}")
        c.append(f"")
        c.append(f"      // Timed runs")
        c.append(f"      std::vector<double> times(runs);")
        c.append(f"      for (int i = 0; i < runs; ++i) {{")
        c.append(f"        q.fill(C, T(0), M * M).wait();")
        c.append(f"        auto ev = kernel_fn(K);")
        c.append(f"        ev.wait();")
        c.append(f"        auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();")
        c.append(f"        auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();")
        c.append(f"        times[i] = (t1 - t0) * 1e-6;")
        c.append(f"      }}")
        c.append(f"      double avg = std::accumulate(times.begin(), times.end(), 0.0) / runs;")
        c.append(f"      double mn  = *std::min_element(times.begin(), times.end());")
        c.append(f"      double sq  = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);")
        c.append(f"      double sd  = std::sqrt(sq / runs - avg * avg);")
        c.append(f"      results.push_back({{label, true, avg, mn, sd, flops / (avg * 1e-3) / 1e9}});")
        c.append(f"    }};")
        c.append(f"")

        # Generate calls for each valid tile size
        for TM in tile_sizes:
            if M % TM != 0:
                continue
            label = f"M{M}_T{TM}"
            bn = f"tsmttsm_{M}_{M}_{TM}_{TM}"
            c.append(f"    bench(\"{label}_basic\",  [&](int k) {{ return {bn}(q, k, A, B, C); }});")
            c.append(f"    bench(\"{label}_lf\",     [&](int k) {{ return {bn}_lf(q, k, A, B, C); }});")
            c.append(f"    bench(\"{label}_sg\",     [&](int k) {{ return {bn}_sg(q, k, A, B, C); }});")
            c.append(f"    bench(\"{label}_sg_lf\",  [&](int k) {{ return {bn}_sg_lf(q, k, A, B, C); }});")
            c.append(f"")

        c.append(f"    sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);")
        c.append(f"    sycl::free(Ah, q); sycl::free(Bh, q); sycl::free(Ch, q);")
        c.append(f"  }}")
        c.append(f"")

    # Print table
    c.append("  // Print results")
    c.append("  const int W = 110;")
    c.append("  std::cout << \"\\n\" << std::string(W, '=') << \"\\n\";")
    c.append("  std::cout << \"BENCHMARK RESULTS\\n\";")
    c.append("  std::cout << std::string(W, '=') << \"\\n\\n\";")
    c.append("  std::cout << std::setw(22) << \"Kernel\"")
    c.append("            << std::setw(10) << \"Correct\"")
    c.append("            << std::setw(12) << \"Avg (ms)\"")
    c.append("            << std::setw(12) << \"Min (ms)\"")
    c.append("            << std::setw(12) << \"Stddev\"")
    c.append("            << std::setw(12) << \"GFLOP/s\" << \"\\n\";")
    c.append("  std::cout << std::string(W, '-') << \"\\n\";")
    c.append("  for (const auto& r : results) {")
    c.append("    std::cout << std::setw(22) << r.name")
    c.append("              << std::setw(10) << (r.correct ? \"PASS\" : \"FAIL\")")
    c.append("              << std::fixed << std::setprecision(4);")
    c.append("    if (r.correct)")
    c.append("      std::cout << std::setw(12) << r.avg_ms")
    c.append("                << std::setw(12) << r.min_ms")
    c.append("                << std::setw(12) << r.stddev_ms")
    c.append("                << std::setprecision(2)")
    c.append("                << std::setw(12) << r.gflops;")
    c.append("    std::cout << \"\\n\";")
    c.append("  }")
    c.append("  std::cout << std::string(W, '=') << \"\\n\";")
    c.append("}")
    c.append("")
    return "\n".join(c)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_stride(M_list, tile_sizes, stride, local_size, sg_size):
    """Check stride divisibility for all generated kernel variants."""
    ok = True

    if stride % local_size != 0:
        print(f"Error: --stride {stride} must be divisible by "
              f"--local-size {local_size}", file=sys.stderr)
        return False

    for M in M_list:
        for TM in tile_sizes:
            if M % TM != 0:
                continue
            mthreads = M // TM
            num_tiles = mthreads * mthreads

            # basic/lf: stride % num_tiles == 0
            if stride % num_tiles != 0:
                print(f"Warning: stride ({stride}) not divisible by num_tiles "
                      f"({num_tiles}) for M={M}, TM={TM}  [basic/lf kernels]",
                      file=sys.stderr)
                ok = False

            # sg/sg_lf: (stride / sg_size) % num_tiles == 0
            if (stride // sg_size) % num_tiles != 0:
                print(f"Warning: stride/sg_size ({stride // sg_size}) not divisible "
                      f"by num_tiles ({num_tiles}) for M={M}, TM={TM}  [sg kernels]",
                      file=sys.stderr)
                ok = False

    return ok


def suggest_stride(M_list, tile_sizes, local_size, sg_size, target_wgs=2048):
    """Compute smallest stride satisfying all constraints with ~target_wgs workgroups."""
    req = local_size
    for M in M_list:
        for TM in tile_sizes:
            if M % TM != 0:
                continue
            mthreads = M // TM
            num_tiles = mthreads * mthreads
            req = lcm(req, num_tiles)
            req = lcm(req, sg_size * num_tiles)

    # req is in threads; ensure it is also a multiple of local_size
    req_full = lcm(req, local_size)
    # We want roughly local_size * target_wgs total threads
    target_threads = local_size * target_wgs
    n = max(1, (target_threads + req_full - 1) // req_full)
    return n * req_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SYCL TSMTTSM kernels (v2): basic, lf, sg, sg_lf")
    parser.add_argument("--M", type=int, nargs="+", default=[4, 8, 16],
                        help="Matrix sizes (default: 4 8 16)")
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[2, 4],
                        help="Tile sizes TM=TN (default: 2 4)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Total thread count (auto-computed if omitted)")
    parser.add_argument("--local-size", type=int, default=256,
                        help="Workgroup size for nd_range kernels (default: 256)")
    parser.add_argument("--sg-size", type=int, default=32,
                        help="Sub-group size: 32 for NVIDIA, 64 for AMD (default: 32)")
    parser.add_argument("--unroll", type=int, default=1,
                        help="K-loop unroll factor (default: 1)")
    parser.add_argument("--with-benchmark", action="store_true",
                        help="Append benchmark infrastructure")
    args = parser.parse_args()

    # Validate sg_size divides local_size
    if args.local_size % args.sg_size != 0:
        print(f"Error: --local-size ({args.local_size}) must be a multiple of "
              f"--sg-size ({args.sg_size})", file=sys.stderr)
        sys.exit(1)

    # Auto-compute stride if not given
    if args.stride is None:
        # Target ~2048 workgroups — enough to saturate a large GPU
        args.stride = suggest_stride(
            args.M, args.tile_sizes, args.local_size, args.sg_size,
            target_wgs=2048)
    if not validate_stride(args.M, args.tile_sizes,
                           args.stride, args.local_size, args.sg_size):
        sys.exit(1)

    print("#pragma once")
    print("")
    print("#include <sycl/sycl.hpp>")
    print("")
    print("/* Auto-generated SYCL kernels for TSMTTSM (A^T * B = C)")
    print(" * Generated by sycl_codegen_new.py")
    print(" *")
    print(" * Variants per (M, TM):")
    print(" *   basic   - sycl::range, K-loop, global atomics")
    print(" *   lf      - basic + double-buffered prefetch (leap-frog)")
    print(" *   sg      - nd_range, sub_group reduce_over_group, leader atomic")
    print(" *   sg_lf   - sg + double-buffered prefetch")
    print(" *")
    print(f" * stride={args.stride}  local_size={args.local_size}  "
          f"sg_size={args.sg_size}  unroll={args.unroll}")
    print(" */")
    print("")

    for M in args.M:
        print(f"// {'=' * 60}")
        print(f"// M = {M}")
        print(f"// {'=' * 60}")
        print("")
        for TM in args.tile_sizes:
            if M % TM != 0:
                continue
            N = M
            TN = TM
            print(generate_basic_kernel(
                M, N, TM, TN, args.stride,
                leap_frog=False, unroll=args.unroll))
            print(generate_basic_kernel(
                M, N, TM, TN, args.stride,
                leap_frog=True, unroll=args.unroll))
            print(generate_sg_kernel(
                M, N, TM, TN, args.stride, args.local_size, args.sg_size,
                leap_frog=False, unroll=args.unroll))
            print(generate_sg_kernel(
                M, N, TM, TN, args.stride, args.local_size, args.sg_size,
                leap_frog=True, unroll=args.unroll))

    if args.with_benchmark:
        print(generate_benchmark(
            args.M, args.tile_sizes, args.stride, args.local_size, args.sg_size))

    print("// End of generated code")


if __name__ == "__main__":
    main()
