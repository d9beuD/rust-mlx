//! Credited upstream gate/up fusion; unsupported quantization uses gather_qmm.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype};
use std::{cell::RefCell, collections::HashMap};
thread_local! { static KERNELS:RefCell<HashMap<(i32,i32),Kernel>>=RefCell::new(HashMap::new()); }
pub(crate) fn gate_up(
    up: &Linear,
    gate: &Linear,
    x: &Array,
    ids: &Array,
) -> Result<Option<(Array, Array)>> {
    let (Some(u), Some(g)) = (&up.quant, &gate.quant) else {
        return Ok(None);
    };
    if x.ndim() != 3
        || x.shape()[0] != 1
        || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        || ![4, 5].contains(&u.bits)
        || ![32, 64, 128].contains(&u.group_size)
        || u.bits != g.bits
        || u.group_size != g.group_size
        || up.bias.is_some()
        || gate.bias.is_some()
    {
        return Ok(None);
    }
    let (Some(us), Some(ub), Some(gs), Some(gb)) =
        (&up.scales, &up.biases, &gate.scales, &gate.biases)
    else {
        return Ok(None);
    };
    if [us, ub, gs, gb].iter().any(|a| a.dtype() != x.dtype()) {
        return Ok(None);
    }
    let (b, t, k) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    ensure!(
        ids.ndim() == 3 && ids.shape()[..2] == [b, t] && ids.shape()[2] > 0,
        "invalid route shape"
    );
    if up.weight.ndim() != 3 || up.weight.shape() != gate.weight.shape() {
        return Ok(None);
    }
    let n = up.weight.shape()[1];
    if k % 512 != 0
        || n % 8 != 0
        || u.group_size % 16 != 0
        || up.weight.dtype() != Dtype::Uint32
        || gate.weight.dtype() != Dtype::Uint32
    {
        return Ok(None);
    }
    ensure!(
        up.weight.shape()[2] as i64 * 32 == k as i64 * u.bits as i64
            && us.shape() == [up.weight.shape()[0], n, k / u.group_size]
            && ub.shape() == us.shape()
            && gs.shape() == us.shape()
            && gb.shape() == us.shape(),
        "malformed affine expert weights"
    );
    let top = ids.shape()[2];
    KERNELS.with(|kernels| -> Result<_> {
        let mut kernels = kernels.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = kernels.entry((u.bits, u.group_size))
        {
            let (source, header) = if u.bits == 4 {
                (
                    include_str!("../kernels/moe_gate_up4.metal"),
                    include_str!("../kernels/moe_gate_up4.h"),
                )
            } else {
                (
                    include_str!("../kernels/moe_gate_up5.metal"),
                    include_str!("../kernels/moe_gate_up5.h"),
                )
            };
            e.insert(Kernel::with_header(
                &format!("rust_mlx_moe_gate_up{}_g{}", u.bits, u.group_size),
                &[
                    "x",
                    "indices",
                    "up_w",
                    "up_scales",
                    "up_biases",
                    "gate_w",
                    "gate_scales",
                    "gate_biases",
                ],
                &["up_y", "gate_y"],
                source,
                &header.replace("GROUP_SIZE", &u.group_size.to_string()),
            )?);
        }
        let x = x.contiguous()?;
        let ids = ids.as_dtype(Dtype::Int32)?.contiguous()?;
        let shape = [b, t, top, n];
        let mut out = kernels[&(u.bits, u.group_size)].launch(Launch {
            inputs: &[&x, &ids, &up.weight, us, ub, &gate.weight, gs, gb],
            templates: &[
                Template::Dtype("T", x.dtype()),
                Template::Int("VERIFY_T", t),
                Template::Int("K_SIZE", k),
                Template::Int("N_SIZE", n),
                Template::Int("TOP_K", top),
                Template::Int("GROUP_SIZE", u.group_size),
            ],
            outputs: &[(&shape, x.dtype()), (&shape, x.dtype())],
            grid: [32, 2 * (n / 8), b * t * top],
            group: [32, 2, 1],
        })?;
        let gate = out.pop().unwrap();
        let up = out.pop().unwrap();
        Ok(Some((gate.expand_dims(-2)?, up.expand_dims(-2)?)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Quantization;
    use mlx_rs::ops;
    fn linear(bits: i32, group: i32, phase: f32) -> Linear {
        let w = Array::from_iter(
            (0..4 * 32 * 2560).map(|i| ((i as f32 * 0.17 + phase).sin()) * 0.05),
            &[4, 32, 2560],
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
    }
    fn exact(a: &Array, b: &Array) {
        let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
    }
    #[test]
    fn fused_matches_native_gather_for_repeated_and_distinct_routes() {
        for bits in [4, 5] {
            for group in [32, 64, 128] {
                let up = linear(bits, group, 0.1);
                let gate = linear(bits, group, 0.7);
                for t in [1, 2, 4, 7] {
                    let x = Array::from_iter(
                        (0..t * 2560).map(|i| (i as f32 * 0.11).cos()),
                        &[1, t, 2560],
                    )
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                    let ids = Array::from_iter((0..t * 2).map(|i| (i + 1) % 4), &[1, t, 2]);
                    let xe = x.expand_dims_axes(&[-2, -3]).unwrap();
                    let (g, u) = gate_up(&up, &gate, &x, &ids).unwrap().unwrap();
                    for (l, a) in [(&up, &u), (&gate, &g)] {
                        let reference = ops::gather_qmm(
                            &xe,
                            &l.weight,
                            l.scales.as_ref().unwrap(),
                            l.biases.as_ref(),
                            None,
                            &ids,
                            true,
                            group,
                            bits,
                            false,
                        )
                        .unwrap();
                        exact(a, &reference);
                    }
                }
            }
        }
    }
}
