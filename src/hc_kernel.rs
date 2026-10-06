//! Restricted exact MLX down/inject reductions in one dispatch, credited upstream.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::Result;
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp};
use std::cell::{Cell, RefCell};
thread_local! {static KERNEL:RefCell<Option<Kernel>>=const{RefCell::new(None)};static ENABLED:Cell<bool>=Cell::new(default_enabled());}
fn default_enabled() -> bool {
    std::env::var("RUST_MLX_HC_PROJECTION")
        .map(|value| !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        .unwrap_or(true)
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

pub fn set_enabled(value: bool) {
    ENABLED.with(|x| x.set(value));
}
pub(crate) fn project(down: &Linear, inject: &Linear, x: &Array) -> Result<Option<(Array, Array)>> {
    if x.ndim() != 3
        || x.shape()[2] != 10240
        || x.dtype() != Dtype::Bfloat16
        || !(1..=16).contains(&(x.shape()[0] * x.shape()[1]))
        || (x.shape()[1] > 1 && !crate::verification::rows())
    {
        return Ok(None);
    }
    let (Some(dq), Some(iq)) = (&down.quant, &inject.quant) else {
        return Ok(None);
    };
    if dq.bits != iq.bits
        || dq.group_size != iq.group_size
        || dq.mode != "affine"
        || iq.mode != "affine"
        || ![4, 5, 6, 8].contains(&dq.bits)
        || ![32, 64].contains(&dq.group_size)
        || down.bias.is_some()
        || inject.bias.is_some()
    {
        return Ok(None);
    }
    for (l, n) in [(down, 320), (inject, 4)] {
        if l.weight.shape() != [n, 10240 * dq.bits / 32] || l.weight.dtype() != Dtype::Uint32 {
            return Ok(None);
        }
        let (Some(sc), Some(bs)) = (&l.scales, &l.biases) else {
            return Ok(None);
        };
        if sc.shape() != [n, 10240 / dq.group_size]
            || bs.shape() != sc.shape()
            || sc.dtype() != x.dtype()
            || bs.dtype() != x.dtype()
        {
            return Ok(None);
        }
    }
    KERNEL.with(|cell| -> Result<_> {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::with_header(
                "rust_mlx_hc_down_inject",
                &[
                    "x", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b",
                ],
                &["combined"],
                include_str!("../kernels/hc_projection.metal"),
                include_str!("../kernels/hc_projection.h"),
            )?);
        }
        let shape = [x.shape()[0], x.shape()[1], 324];
        let out = cell
            .as_ref()
            .expect("initialized kernel")
            .launch(Launch {
                inputs: &[
                    x,
                    &down.weight,
                    down.scales.as_ref().unwrap(),
                    down.biases.as_ref().unwrap(),
                    &inject.weight,
                    inject.scales.as_ref().unwrap(),
                    inject.biases.as_ref().unwrap(),
                ],
                templates: &[
                    Template::Dtype("T", x.dtype()),
                    Template::Int("BITS", dq.bits),
                    Template::Int("K", 10240),
                    Template::Int("GS", dq.group_size),
                ],
                outputs: &[(&shape, x.dtype())],
                grid: [32, 82, x.shape()[0] * x.shape()[1]],
                group: [32, 2, 1],
            })?
            .remove(0);
        Ok(Some((
            out.index((.., .., ..320)),
            out.index((.., .., 320..)),
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Quantization;
    use mlx_rs::ops;
    fn exact(a: &Array, b: &Array) {
        let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
    }
    #[test]
    fn native_fast_and_small_projection_trees_remain_exact() {
        for bits in [4, 5, 6, 8] {
            for group in [32, 64] {
                let make = |n| {
                    let w = Array::from_iter(
                        (0..n * 10240).map(|i| (i as f32 * 0.023).sin() * 0.01),
                        &[n, 10240],
                    )
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                    let (weight, scales, biases) = ops::quantize(w, group, bits).unwrap();
                    Linear {
                        weight,
                        scales: Some(scales),
                        biases: Some(biases),
                        bias: None,
                        quant: Some(Quantization {
                            bits,
                            group_size: group,
                            mode: "affine".into(),
                        }),
                    }
                };
                let down = make(320);
                let inject = make(4);
                for (b, t) in [(1, 1), (1, 4), (1, 8), (8, 1)] {
                    let x = Array::from_iter(
                        (0..b * t * 10240).map(|i| (i as f32 * 0.071).cos()),
                        &[b, t, 10240],
                    )
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                    let (d, i) = crate::verification::with_rows(|| project(&down, &inject, &x))
                        .unwrap()
                        .unwrap();
                    exact(&d, &down.forward_rows(&x).unwrap());
                    exact(&i, &inject.forward_rows(&x).unwrap());
                }
            }
        }
    }
}
