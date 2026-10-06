//! Exact affine short-block projection with shared weight loads, credited upstream.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};
use std::{cell::RefCell, collections::HashMap};
thread_local! {static KERNELS:RefCell<HashMap<(i32,i32,bool),Kernel>>=RefCell::new(HashMap::new());static ENABLED:std::cell::Cell<bool>=std::cell::Cell::new(default_enabled());}
thread_local! {static STREAM_X:std::cell::Cell<bool>=std::cell::Cell::new(env_enabled("RUST_MLX_QMV_STREAM_X",false));}
pub fn stream_x() -> bool {
    STREAM_X.with(std::cell::Cell::get)
}
pub fn set_stream_x(value: bool) {
    STREAM_X.with(|x| x.set(value));
}
thread_local! {static BATCH_ENABLED:std::cell::Cell<bool>=std::cell::Cell::new(env_enabled("RUST_MLX_BATCH_QMV",true));}
pub fn set_batch_enabled(value: bool) {
    BATCH_ENABLED.with(|x| x.set(value));
}
pub fn batch_enabled() -> bool {
    BATCH_ENABLED.with(std::cell::Cell::get)
}
fn default_enabled() -> bool {
    env_enabled("RUST_MLX_VERIFY_QMV", true)
}
fn env_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        .unwrap_or(default)
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|x| x.set(value));
}
pub fn enabled() -> bool {
    ENABLED.with(std::cell::Cell::get)
}
pub(crate) fn project(l: &Linear, x: &Array) -> Result<Option<Array>> {
    if batch_enabled() && x.ndim() == 3 && x.shape()[1] == 1 && (2..=8).contains(&x.shape()[0]) {
        let b = x.shape()[0];
        let flat = x.contiguous()?.reshape(&[1, b, x.shape()[2]])?;
        return project(l, &flat)?
            .map(|y| y.reshape(&[b, 1, y.shape()[2]]).map_err(Into::into))
            .transpose();
    }
    // Measured dispatch: small output matrices and long 4/5/6-bit blocks lose
    // occupancy. The 8-bit vocabulary head benefits at every verifier length.
    if l.weight.ndim() != 2
        || l.weight.shape()[0] < 1024
        || (x.ndim() == 3
            && x.shape()[1] > 4
            && !stream_x()
            && l.quant.as_ref().is_none_or(|q| q.bits != 8))
    {
        return Ok(None);
    }
    project_unfiltered(l, x)
}
fn project_unfiltered(l: &Linear, x: &Array) -> Result<Option<Array>> {
    let Some(q) = &l.quant else { return Ok(None) };
    if q.mode != "affine"
        || x.ndim() != 3
        || !(2..=8).contains(&x.shape()[1])
        || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        || ![4, 5, 6, 8].contains(&q.bits)
        || ![32, 64, 128].contains(&q.group_size)
        || l.weight.ndim() != 2
        || l.weight.dtype() != Dtype::Uint32
    {
        return Ok(None);
    }
    let (Some(sc), Some(bs)) = (&l.scales, &l.biases) else {
        return Ok(None);
    };
    if sc.dtype() != x.dtype() || bs.dtype() != x.dtype() {
        return Ok(None);
    }
    let (b, t, k) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let n = l.weight.shape()[0];
    if b <= 0 || k <= 0 || n <= 0 || k % 512 != 0 || n % 8 != 0 {
        return Ok(None);
    }
    ensure!(
        b > 0
            && l.weight.shape()[1] as i64 * 32 == k as i64 * q.bits as i64
            && sc.shape() == [n, k / q.group_size]
            && sc.shape() == bs.shape(),
        "malformed verifier projection"
    );
    // Large actual-model T6/T8 shaders report invalid device addresses under
    // GPU validation, although their numerical outputs pass. Bound the live
    // thread arrays to four independent positions; their arithmetic/reduction
    // order stays identical. Keep a native singleton tail for odd lengths.
    if t > 4 {
        let mut blocks = Vec::new();
        for start in (0..t).step_by(4) {
            let end = (start + 4).min(t);
            let block = x.index((.., start..end, ..));
            blocks.push(if end - start == 1 {
                l.forward(&block)?
            } else {
                project_unfiltered(l, &block)?.context("unchanged eligible QMV geometry")?
            });
        }
        return Ok(Some(ops::concatenate(&blocks, 1)?));
    }
    let x = x.contiguous()?;
    let key = (q.bits, q.group_size, stream_x());
    KERNELS.with(|kernels| -> Result<_> {
        let mut kernels = kernels.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = kernels.entry(key) {
            let header = include_str!("../kernels/verify_qmv.h")
                .replace("__BITS__", &q.bits.to_string())
                .replace("__GS__", &q.group_size.to_string());
            e.insert(Kernel::with_header(
                &format!(
                    "rust_mlx_verify_qmv{}_g{}_stream{}",
                    q.bits, q.group_size, key.2
                ),
                &["x", "w", "scales", "biases"],
                &["y"],
                if key.2 {
                    include_str!("../kernels/verify_qmv_streamed.metal")
                } else {
                    include_str!("../kernels/verify_qmv.metal")
                },
                &header,
            )?);
        }
        let shape = [b, t, n];
        let mut y = kernels[&key]
            .launch(Launch {
                inputs: &[&x, &l.weight, sc, bs],
                templates: &[
                    Template::Dtype("T", x.dtype()),
                    Template::Int("VERIFY_T", t),
                    Template::Int("K_SIZE", k),
                    Template::Int("N_SIZE", n),
                ],
                outputs: &[(&shape, x.dtype())],
                grid: [32, 2 * (n / 8), b],
                group: [32, 2, 1],
            })?
            .remove(0);
        if let Some(bias) = &l.bias {
            y = y.add(bias)?;
        }
        Ok(Some(y))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Quantization;
    use mlx_rs::ops::{self, indexing::IndexOp};
    fn exact(a: &Array, b: &Array) {
        let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
    }
    #[test]
    fn shared_weights_match_singleton_quantized_projection() {
        set_stream_x(false);
        singleton_cases();
    }
    #[test]
    fn streamed_rows_match_singleton_quantized_projection() {
        set_stream_x(true);
        singleton_cases();
        set_stream_x(false);
    }
    fn singleton_cases() {
        for bits in [4, 5, 6, 8] {
            for group in [32, 64, 128] {
                for (k, n) in [(512, 48), (2560, 640), (10240, 320)] {
                    let w = Array::from_iter(
                        (0..k * n).map(|i| (i as f32 * 0.017).sin() * 0.03),
                        &[n, k],
                    )
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                    let (weight, scales, biases) = ops::quantize(w, group, bits).unwrap();
                    let l = Linear {
                        weight,
                        scales: Some(scales),
                        biases: Some(biases),
                        bias: None,
                        quant: Some(Quantization {
                            bits,
                            group_size: group,
                            mode: "affine".into(),
                        }),
                    };
                    for t in [2, 4, 8] {
                        let x = Array::from_iter(
                            (0..t * k).map(|i| (i as f32 * 0.11).cos()),
                            &[1, t, k],
                        )
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap();
                        let y = project_unfiltered(&l, &x).unwrap().unwrap();
                        let reference = (0..t)
                            .map(|i| l.forward(&x.index((.., i..i + 1, ..))).unwrap())
                            .collect::<Vec<_>>();
                        exact(&y, &ops::concatenate(&reference, 1).unwrap());
                    }
                }
            }
        }
    }
    #[test]
    fn batch_rows_share_weights_without_changing_singleton_reductions() {
        set_batch_enabled(true);
        for bits in [4, 5, 6, 8] {
            let k = 512;
            let n = 1024;
            let w = Array::from_iter((0..k * n).map(|i| (i as f32 * 0.017).sin() * 0.03), &[n, k])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let (weight, scales, biases) = ops::quantize(w, 64, bits).unwrap();
            let l = Linear {
                weight,
                scales: Some(scales),
                biases: Some(biases),
                bias: None,
                quant: Some(Quantization {
                    bits,
                    group_size: 64,
                    mode: "affine".into(),
                }),
            };
            for b in [2, 4, 8] {
                // Strided rows exercise the layout adapter, not just a flat copy.
                let x =
                    Array::from_iter((0..b * 2 * k).map(|i| (i as f32 * 0.11).cos()), &[b, 2, k])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .index((.., 1..2, ..));
                let Some(y) = project(&l, &x).unwrap() else {
                    assert!(b > 4 && bits != 8, "unexpected batch fallback");
                    continue;
                };
                let reference = (0..b)
                    .map(|i| l.forward(&x.index((i..i + 1, .., ..))).unwrap())
                    .collect::<Vec<_>>();
                exact(&y, &ops::concatenate(&reference, 0).unwrap());
            }
        }
        set_batch_enabled(false);
    }
}
