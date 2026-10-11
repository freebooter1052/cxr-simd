# kernels/CMakeLists.txt
# ----------------------
# Build system for NEON kernels, validation suite, and micro-benchmark.
#
# Usage on Raspberry Pi (aarch64 Raspberry Pi OS 64-bit):
#
#   cd kernels
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j4
#
# Then run:
#   ./validate     <- must print "ALL TESTS PASSED"
#   ./bench        <- prints CSV speedup table
#
cmake_minimum_required(VERSION 3.16)
project(cxr_simd_kernels CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Detect ARM / NEON ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Target: aarch64 -- enabling native NEON")
    set(ARCH_FLAGS "-march=armv8-a+simd -ffast-math")
else()
    message(STATUS "Target: scalar fallback only")
    set(ARCH_FLAGS "")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${ARCH_FLAGS}")

# ---- Shared kernel sources ----
set(KERNEL_SRCS
    neon_kernels.cpp
    scalar_ref/scalar_kernels.cpp
)

# ---- Validate binary ----
add_executable(validate
    ${KERNEL_SRCS}
    tests/validate_kernels.cpp
)
target_include_directories(validate PRIVATE . scalar_ref)

# ---- Bench binary ----
add_executable(bench
    ${KERNEL_SRCS}
    tests/bench_kernels.cpp
)
target_include_directories(bench PRIVATE . scalar_ref)

# ---- Shared library (for Python ctypes bridge) ----
add_library(neon_kernels SHARED ${KERNEL_SRCS})
target_include_directories(neon_kernels PRIVATE . scalar_ref)
set_target_properties(neon_kernels PROPERTIES
    OUTPUT_NAME "neon_kernels"
)
