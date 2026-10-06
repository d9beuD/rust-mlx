#!/bin/sh
# Process-local diagnostic instrumentation, never a throughput benchmark.
set -eu
export MTL_SHADER_VALIDATION=1
export MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1
export MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1
export MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1
cargo test --release --lib -- --test-threads=1
cargo test --release --test native_kernels -- --test-threads=1
cargo test --release --test gdn_compile -- --test-threads=1
cargo test --release --test sorted_moe -- --test-threads=1
printf 'METAL_VALIDATION_PASSED\n'
