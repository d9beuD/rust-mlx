#!/bin/sh
set -eu
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release
printf 'QUALITY_CHECKS_PASSED\n'
