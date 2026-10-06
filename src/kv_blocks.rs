//! Opt-in fixed-block KV backing; MLX functional updates preserve shared snapshots.
use std::cell::Cell;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|v| v.set(value));
}
pub fn calls() -> u64 {
    CALLS.with(Cell::get)
}
pub(crate) fn record() {
    CALLS.with(|v| v.set(v.get().wrapping_add(1)));
}
pub const BLOCK: i32 = 256;
