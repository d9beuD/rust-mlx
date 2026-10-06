//! Process-local research taps. No tensors leave the owning MLX thread.
use mlx_rs::Array;
use std::cell::{Cell, RefCell};
#[derive(Default)]
struct Capture {
    active: bool,
    calls: usize,
    arrays: Vec<(String, Array)>,
}
thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; static CAPTURE: RefCell<Capture> = RefCell::new(Capture::default()); }
pub fn start() {
    ACTIVE.with(|v| v.set(true));
    CAPTURE.with(|c| {
        *c.borrow_mut() = Capture {
            active: true,
            ..Capture::default()
        }
    });
}
pub fn finish() -> (usize, Vec<(String, Array)>) {
    ACTIVE.with(|v| v.set(false));
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        c.active = false;
        (c.calls, std::mem::take(&mut c.arrays))
    })
}
pub(crate) fn record(x: &Array, routed: &Array, ids: &Array, scores: &Array) {
    if !ACTIVE.with(Cell::get) {
        return;
    }
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.active {
            return;
        }
        let layer = c.calls;
        c.calls += 1;
        if [0, 23, 47].contains(&layer) {
            for (key, value) in [
                ("input", x),
                ("routed", routed),
                ("ids", ids),
                ("scores", scores),
            ] {
                c.arrays
                    .push((format!("layer{layer}.{key}"), value.clone()));
            }
        }
    });
}

pub(crate) fn record_epilogue(
    routed: &Array,
    shared: &Array,
    factor: &Array,
    residual: &Array,
    gate: &Array,
) {
    if !ACTIVE.with(Cell::get) {
        return;
    }
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.active || c.calls == 0 {
            return;
        }
        let layer = c.calls - 1;
        if [0, 23, 47].contains(&layer) {
            for (key, value) in [
                ("down", routed),
                ("shared", shared),
                ("factor", factor),
                ("residual", residual),
                ("injection", gate),
            ] {
                c.arrays
                    .push((format!("layer{layer}.{key}"), value.clone()));
            }
        }
    });
}
