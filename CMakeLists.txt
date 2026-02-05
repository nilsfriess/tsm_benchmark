cmake_minimum_required(VERSION 3.16)
project(tsm_benchmark LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)

add_executable(main main.cc)
target_precompile_headers(main PRIVATE <sycl/sycl.hpp>)
add_sycl_to_target(TARGET main)

add_executable(main_codegen main_codegen.cc)
target_precompile_headers(main_codegen PRIVATE <sycl/sycl.hpp>)
add_sycl_to_target(TARGET main_codegen)
