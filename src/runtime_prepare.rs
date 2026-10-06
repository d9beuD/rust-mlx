//! Bounded, owning-thread preparation policies, experimental and off by default.
use anyhow::Result;
use mlx_rs::Array;
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Positions {
    batch: i32,
    rows: i32,
    offset: i32,
    stride: i32,
}
thread_local! {
    static CONFIG: Cell<bool> = const { Cell::new(false) };
    static CONFIG_HITS: Cell<usize> = const { Cell::new(0) };
    static PLE: Cell<bool> = const { Cell::new(false) };
    static ROPE: Cell<bool> = const { Cell::new(false) };
    static PLE_CALLS: Cell<usize> = const { Cell::new(0) };
    static POSITION_HITS: Cell<usize> = const { Cell::new(0) };
    static IDS: RefCell<VecDeque<(Positions,Array)>> = const { RefCell::new(VecDeque::new()) };
}
pub fn configure(mode: Option<&str>, enabled: bool) {
    set_ple(enabled && matches!(mode, Some("ple-prepare" | "runtime-prepare")));
    set_rope(enabled && matches!(mode, Some("rope-ids" | "runtime-prepare")));
    CONFIG.with(|v| v.set(enabled && matches!(mode, Some("config-reuse" | "runtime-prepare"))));
}
pub fn config_enabled() -> bool {
    CONFIG.with(Cell::get)
}
pub fn config_hits() -> usize {
    CONFIG_HITS.with(Cell::get)
}
pub fn record_config_hit() {
    CONFIG_HITS.with(|v| v.set(v.get().wrapping_add(1)));
}
pub fn stats() -> [usize; 3] {
    [ple_calls(), position_hits(), config_hits()]
}
pub fn set_ple(enabled: bool) {
    PLE.with(|v| v.set(enabled));
}
pub fn ple_enabled() -> bool {
    PLE.with(Cell::get)
}
pub fn set_rope(enabled: bool) {
    ROPE.with(|v| v.set(enabled));
}
pub fn ple_calls() -> usize {
    PLE_CALLS.with(Cell::get)
}
pub fn position_hits() -> usize {
    POSITION_HITS.with(Cell::get)
}
pub fn record_ple() {
    PLE_CALLS.with(|v| v.set(v.get().wrapping_add(1)));
}

fn uncached(k: Positions) -> Array {
    Array::from_iter(
        (0..k.batch).flat_map(|_| (k.offset..k.offset + k.rows).map(|p| p * k.stride)),
        &[k.batch, k.rows],
    )
}
/// Share immutable integer position IDs across layers in a narrow decode block.
/// Eight entries with b/t <=8 retain at most2KiB. Offset/stride are part of every
/// key, so verifier rollback and independent compatible batches cannot collide.
pub fn positions(batch: i32, rows: i32, offset: i32, stride: i32) -> Result<Array> {
    let key = Positions {
        batch,
        rows,
        offset,
        stride,
    };
    if !ROPE.with(Cell::get) || !(1..=8).contains(&batch) || !(1..=8).contains(&rows) {
        return Ok(uncached(key));
    }
    IDS.with(|ids| {
        let mut ids = ids.borrow_mut();
        if let Some(index) = ids.iter().position(|(k, _)| *k == key) {
            let value = ids.remove(index).expect("existing entry");
            let result = value.1.clone();
            ids.push_back(value);
            POSITION_HITS.with(|v| v.set(v.get().wrapping_add(1)));
            return Ok(result);
        }
        if ids.len() == 8 {
            ids.pop_front();
        }
        let result = uncached(key);
        ids.push_back((key, result.clone()));
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn position_ids_preserve_offsets_strides_batches_and_rollback() {
        set_rope(true);
        for offset in [0, 1, 4, 100, 2052, 2056, 2060, 2064, 2068, 2072, 2052, 1, 0] {
            for batch in [1, 8] {
                for rows in [1, 4, 8] {
                    for stride in [1, 4] {
                        let a = positions(batch, rows, offset, stride).unwrap();
                        let b = positions(batch, rows, offset, stride).unwrap();
                        let expected = uncached(Positions {
                            batch,
                            rows,
                            offset,
                            stride,
                        });
                        mlx_rs::transforms::eval([&a, &b, &expected]).unwrap();
                        assert_eq!(a.as_slice::<i32>(), expected.as_slice::<i32>());
                        assert_eq!(b.as_slice::<i32>(), expected.as_slice::<i32>());
                    }
                }
            }
        }
        assert!(position_hits() > 0);
        IDS.with(|ids| assert!(ids.borrow().len() <= 8));
        set_rope(false);
    }
}
