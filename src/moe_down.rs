//! Exact experimental MTPLX down/reduction and selected lossless word packing.
use crate::{
    hybrid::HybridModel,
    metal::{Kernel, Launch},
    weights::Linear,
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype};
use std::cell::{Cell, RefCell};
thread_local! {
    static MODE:Cell<u8>=const {Cell::new(0)};
    static CALLS:Cell<usize>=const {Cell::new(0)};
    static KERNELS:RefCell<[Option<Kernel>;3]>=const {RefCell::new([None,None,None])};
}
pub fn configure(kind: Option<&str>, enabled: bool) {
    MODE.with(|m| {
        m.set(if !enabled {
            0
        } else {
            match kind {
                Some("down-tail") => 1,
                Some("down-packed") => 2,
                Some("down-packed-vector") => 3,
                _ => 0,
            }
        })
    });
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}
/// Only native router-produced indices reach this function. The MoE owns the
/// bounds invariant (512 expert IDs) and all arrays remain on its MLX thread.
pub(crate) fn reduce(
    l: &Linear,
    x: &Array,
    ids: &Array,
    scores: &Array,
    packed: Option<&Array>,
) -> Result<Option<Array>> {
    let mode = MODE.with(Cell::get);
    if mode == 0 || (mode >= 2 && packed.is_none()) {
        return Ok(None);
    }
    if x.ndim() != 5
        || x.shape()[0] != 1
        || !(1..=8).contains(&x.shape()[1])
        || x.shape()[2..] != [10, 1, 640]
        || x.dtype() != Dtype::Bfloat16
        || l.weight.shape() != [512, 2560, 80]
        || l.weight.dtype() != Dtype::Uint32
        || l.quant
            .as_ref()
            .is_none_or(|q| q.bits != 4 || q.group_size != 64 || q.mode != "affine")
        || l.bias.is_some()
        || ids.shape() != [1, x.shape()[1], 10]
        || ids.dtype() != Dtype::Uint32
        || scores.shape() != ids.shape()
        || scores.dtype() != Dtype::Bfloat16
    {
        return Ok(None);
    }
    let sc = l.scales.as_ref().context("missing down scales")?;
    let bs = l.biases.as_ref().context("missing down biases")?;
    if sc.shape() != [512, 2560, 10]
        || bs.shape() != sc.shape()
        || sc.dtype() != Dtype::Bfloat16
        || bs.dtype() != sc.dtype()
    {
        return Ok(None);
    }
    let packed = if mode >= 2 { packed } else { None };
    if packed.is_some_and(|w| w.shape() != [512, 640, 80, 4] || w.dtype() != Dtype::Uint32) {
        return Ok(None);
    }
    KERNELS.with(|kernels| {
        let mut kernels = kernels.borrow_mut();
        let slot = &mut kernels[if mode == 3 {
            2
        } else {
            usize::from(packed.is_some())
        }];
        if slot.is_none() {
            eprintln!("Powered by MTPLX — https://github.com/youssofal/mtplx");
            let (name, header, source) = if mode == 3 {
                (
                    "rust_mlx_selected_down_vector",
                    include_str!("../kernels/moe_down_vector.h"),
                    include_str!("../kernels/moe_down_vector.metal"),
                )
            } else if packed.is_some() {
                (
                    "rust_mlx_selected_down_packed",
                    include_str!("../kernels/moe_down_packed.h"),
                    include_str!("../kernels/moe_down_packed.metal"),
                )
            } else {
                (
                    "rust_mlx_native_down_tail",
                    include_str!("../kernels/moe_down_tail.h"),
                    include_str!("../kernels/moe_down_tail.metal"),
                )
            };
            *slot = Some(Kernel::with_header(
                name,
                &[
                    "routed_h",
                    "weights",
                    "scales",
                    "biases",
                    "expert_ids",
                    "route_scores",
                ],
                &["routed_down"],
                source,
                header,
            )?);
        }
        let x = x.contiguous()?;
        let ids = ids.contiguous()?;
        let scores = scores.contiguous()?;
        let rows = x.shape()[1];
        let y = slot
            .as_ref()
            .unwrap()
            .launch(Launch {
                inputs: &[&x, packed.unwrap_or(&l.weight), sc, bs, &ids, &scores],
                templates: &[],
                outputs: &[(&[1, rows, 2560], Dtype::Bfloat16)],
                grid: [20480, rows, 1],
                group: [64, 1, 1],
            })?
            .remove(0);
        CALLS.with(|v| v.set(v.get().wrapping_add(1)));
        Ok(Some(y))
    })
}
/// Predetermined three target banks bound the additional resident allocation.
/// Original contiguous banks remain intact for a genuine native control/fallback.
pub fn prepare_selected(m: &mut HybridModel) -> Result<serde_json::Value> {
    let started = std::time::Instant::now();
    let mut bytes = 0;
    for i in [0, 23, 47] {
        let l = &mut m.layers[i].moe;
        ensure!(
            l.down.weight.shape() == [512, 2560, 80],
            "unsupported packed bank"
        );
        let p = l
            .down
            .weight
            .reshape(&[512, 640, 4, 80])?
            .transpose_axes(&[0, 1, 3, 2])?
            .contiguous()?;
        p.eval()?;
        bytes += p.nbytes();
        l.down_packed = Some(p);
    }
    Ok(
        serde_json::json!({"layers":[0,23,47],"additional_code_bytes":bytes,"preparation_seconds":started.elapsed().as_secs_f64(),"lossless":true,"scale_bias":"original native owners","native_control":"original evaluated row-contiguous codes retained; no strided reconstruction","scope":"selected3 banks only; packed and control arms retain same additional allocation"}),
    )
}
