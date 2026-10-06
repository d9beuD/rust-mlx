//! Exact greedy-only8-bit head experiment, with full native-logit fallback.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, ops::indexing};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
    static PLANS: RefCell<HashMap<i32, (Kernel, Kernel)>> = RefCell::new(HashMap::new());
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

pub fn greedy(l: &Linear, x: &Array) -> Result<Array> {
    if enabled()
        && let Some((_, _, ids)) = project(l, x)?
    {
        return Ok(ids);
    }
    Ok(indexing::argmax_axis(l.forward_rows(x)?, -1, false)?)
}
/// Component oracle access without materializing the discarded token reduction.
pub fn diagnostic_partials(l: &Linear, x: &Array) -> Result<Option<(Array, Array)>> {
    Ok(project(l, x)?.map(|(maxima, indices, _)| (maxima, indices)))
}
fn project(l: &Linear, x: &Array) -> Result<Option<(Array, Array, Array)>> {
    let Some(q) = &l.quant else { return Ok(None) };
    let (Some(sc), Some(bs)) = (&l.scales, &l.biases) else {
        return Ok(None);
    };
    if q.mode != "affine"
        || q.bits != 8
        || ![32, 64, 128].contains(&q.group_size)
        || l.bias.is_some()
        || x.ndim() != 3
        || x.dtype() != Dtype::Bfloat16
        || l.weight.ndim() != 2
        || l.weight.dtype() != Dtype::Uint32
        || sc.dtype() != x.dtype()
        || bs.dtype() != x.dtype()
        || !(1..=4).contains(&x.shape()[1])
        || x.shape()[0] < 1
    {
        return Ok(None);
    }
    let (b, t, k, n) = (
        x.shape()[0],
        x.shape()[1],
        x.shape()[2],
        l.weight.shape()[0],
    );
    if k < 512
        || k % 512 != 0
        || n < 8
        || n % 8 != 0
        || l.weight.shape()[1] != k / 4
        || sc.shape() != [n, k / q.group_size]
        || bs.shape() != sc.shape()
        || l.weight.size() as i64 * 4 > i32::MAX as i64
        || x.size() as i64 > i32::MAX as i64
        || b as i64 * t as i64 * n as i64 > i32::MAX as i64
    {
        return Ok(None);
    }
    let x = x.contiguous()?;
    PLANS.with(|plans| -> Result<_> {
        let mut plans = plans.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = plans.entry(q.group_size) {
            let header = include_str!("../kernels/verify_qmv.h")
                .replace("__BITS__", "8")
                .replace("__GS__", &q.group_size.to_string());
            let projection = Kernel::with_header(
                &format!("rust_mlx_greedy_head_g{}", q.group_size),
                &["x", "w", "scales", "biases"],
                &["maxima", "indices"],
                include_str!("../kernels/greedy_head.metal"),
                &header,
            )?;
            let reduction = Kernel::new(
                "rust_mlx_greedy_reduce",
                &["maxima", "indices"],
                &["tokens"],
                include_str!("../kernels/greedy_reduce.metal"),
            )?;
            entry.insert((projection, reduction));
        }
        let (projection, reduction) = plans.get(&q.group_size).context("inserted head plan")?;
        let shape = [b * t, n / 8];
        let partials = projection.launch(Launch {
            inputs: &[&x, &l.weight, sc, bs],
            templates: &[
                Template::Dtype("T", x.dtype()),
                Template::Int("VERIFY_T", t),
                Template::Int("K_SIZE", k),
                Template::Int("N_SIZE", n),
            ],
            outputs: &[(&shape, Dtype::Float32), (&shape, Dtype::Uint32)],
            grid: [32, 2 * (n / 8), b],
            group: [32, 2, 1],
        })?;
        let out = reduction
            .launch(Launch {
                inputs: &[&partials[0], &partials[1]],
                templates: &[Template::Int("PARTIALS", n / 8)],
                outputs: &[(&[b, t], Dtype::Uint32)],
                grid: [256, b * t, 1],
                group: [256, 1, 1],
            })?
            .remove(0);
        CALLS.with(|v| v.set(v.get().wrapping_add(1)));
        Ok(Some((partials[0].clone(), partials[1].clone(), out)))
    })
}
