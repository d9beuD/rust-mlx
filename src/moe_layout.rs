//! Lossless expert gate/up row concatenation, using native MLX gather arithmetic.
use crate::{
    hybrid::{HybridModel, MoE},
    weights::{Linear, Weights},
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{ops, ops::indexing::IndexOp};
use std::cell::Cell;
thread_local! {
    static ENABLED: Cell<bool> = const {Cell::new(false)};
    static CALLS: Cell<usize> = const {Cell::new(0)};
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|v| v.set(value));
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}
pub(crate) fn record_call() {
    CALLS.with(|v| v.set(v.get().wrapping_add(1)));
}
pub fn prepare(m: &mut MoE) -> Result<()> {
    let (g, u) = (&m.gate, &m.up);
    let q = g.quant.as_ref().context("gate quantization missing")?;
    let qu = u.quant.as_ref().context("up quantization missing")?;
    ensure!(
        g.weight.ndim() == 3
            && g.weight.shape() == u.weight.shape()
            && q.mode == "affine"
            && qu.mode == q.mode
            && qu.bits == q.bits
            && qu.group_size == q.group_size
            && g.bias.is_none()
            && u.bias.is_none(),
        "incompatible native gate/up layout"
    );
    let concat = |a: &Option<mlx_rs::Array>, b: &Option<mlx_rs::Array>| -> Result<_> {
        let a = a.as_ref().context("gate metadata missing")?;
        let b = b.as_ref().context("up metadata missing")?;
        ensure!(
            a.shape() == b.shape() && a.dtype() == b.dtype(),
            "gate/up metadata mismatch"
        );
        Ok(ops::concatenate(&[a, b], 1)?)
    };
    let l = Linear {
        weight: ops::concatenate(&[&g.weight, &u.weight], 1)?,
        scales: Some(concat(&g.scales, &u.scales)?),
        biases: Some(concat(&g.biases, &u.biases)?),
        bias: None,
        quant: Some(q.clone()),
    };
    mlx_rs::transforms::eval([
        &l.weight,
        l.scales.as_ref().unwrap(),
        l.biases.as_ref().unwrap(),
    ])?;
    m.gate_up = Some(l);
    Ok(())
}
/// Replace original bank ownership by views of the same packed allocation.
/// Reference gather remains numerically available, but may copy strided views;
/// performance comparisons must load the original layout in a separate arm.
pub fn prepare_model(m: &mut HybridModel, w: &mut Weights) -> Result<serde_json::Value> {
    let started = std::time::Instant::now();
    let mut bytes = 0;
    for (i, layer) in m.layers.iter_mut().enumerate() {
        prepare(&mut layer.moe)?;
        let l = layer
            .moe
            .gate_up
            .as_ref()
            .context("prepared gate/up missing")?;
        let n = l.weight.shape()[1] / 2;
        let view = |start, end| -> Result<Linear> {
            Ok(Linear {
                weight: l.weight.index((.., start..end, ..)),
                scales: Some(l.scales.as_ref().context("scales missing")?.index((
                    ..,
                    start..end,
                    ..,
                ))),
                biases: Some(l.biases.as_ref().context("biases missing")?.index((
                    ..,
                    start..end,
                    ..,
                ))),
                bias: None,
                quant: l.quant.clone(),
            })
        };
        let g = view(0, n)?;
        let u = view(n, 2 * n)?;
        bytes += l.weight.nbytes()
            + l.scales.as_ref().unwrap().nbytes()
            + l.biases.as_ref().unwrap().nbytes();
        layer.moe.gate = g;
        layer.moe.up = u;
        for module in ["gate_proj", "up_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                w.tensors.remove(&format!(
                    "language_model.model.layers.{i}.mlp.switch_mlp.{module}.{suffix}"
                ));
            }
        }
    }
    Ok(
        serde_json::json!({"layers":m.layers.len(),"bytes":bytes,"preparation_seconds":started.elapsed().as_secs_f64(),
        "lossless":true,"arithmetic":"native gather_qmm; code/scales/bias row concatenation only",
        "comparison":"original-layout controls must be separately loaded; strided-view fallback is numerical reference only"}),
    )
}
