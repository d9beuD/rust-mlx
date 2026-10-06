//! Research-only activation moments for resident affine weight calibration.
//! CPU statistics stay on the owning MLX thread; tensors are never sent elsewhere.
use crate::{
    hybrid::{HybridAttention, HybridModel},
    weights::Linear,
};
use anyhow::{Result, ensure};
use mlx_rs::Dtype;
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap},
};
thread_local! {
    static ACTIVE:Cell<bool>=const {Cell::new(false)};
    static NAMES:RefCell<HashMap<usize,String>>=RefCell::new(HashMap::new());
    static STATS:RefCell<BTreeMap<String,ActivationMoments>>=const {RefCell::new(BTreeMap::new())};
}
#[derive(Serialize, Deserialize)]
pub struct ActivationMoments {
    pub rows: usize,
    pub sum_squares: Vec<f64>,
}
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}
pub(crate) fn record(l: &Linear, x: &mlx_rs::Array) -> Result<()> {
    let name = NAMES.with(|n| n.borrow().get(&(l as *const Linear as usize)).cloned());
    let Some(name) = name else { return Ok(()) };
    ensure!(x.ndim() >= 2, "calibration input rank");
    let k = x.shape()[x.ndim() - 1];
    ensure!(k > 0 && x.size() > 0, "empty calibration input");
    let count = x.size() / k as usize;
    let x = x.as_dtype(Dtype::Float32)?.reshape(&[-1, k])?;
    let sum = x.multiply(&x)?.sum_axis(0, false)?.contiguous()?;
    sum.eval()?;
    let values = sum.as_slice::<f32>();
    ensure!(
        values.iter().all(|v| v.is_finite() && *v >= 0.),
        "nonfinite activation moment"
    );
    STATS.with(|s| -> Result<()> {
        let mut s = s.borrow_mut();
        let entry = s.entry(name).or_insert_with(|| ActivationMoments {
            rows: 0,
            sum_squares: vec![0.; k as usize],
        });
        ensure!(
            entry.sum_squares.len() == values.len(),
            "calibration projection shape changed"
        );
        entry.rows += count;
        for (sum, value) in entry.sum_squares.iter_mut().zip(values) {
            *sum += *value as f64;
        }
        Ok(())
    })
}
/// Register addresses only while the model is immutably borrowed by this call.
/// Addresses are opaque keys, never dereferenced; they cannot outlive this scope.
pub fn capture(
    m: &HybridModel,
    run: impl FnOnce(&HybridModel) -> Result<()>,
) -> Result<BTreeMap<String, ActivationMoments>> {
    ensure!(!active(), "nested model calibration");
    let mut names = HashMap::new();
    let mut register = |l: &Linear, p: String| {
        if l.weight.ndim() == 2
            && l.weight.shape()[0] >= 512
            && l.quant
                .as_ref()
                .is_some_and(|q| q.mode == "affine" && q.bits > 4)
            && l.scales
                .as_ref()
                .is_some_and(|s| s.dtype() == Dtype::Bfloat16)
        {
            names.insert(l as *const Linear as usize, p);
        }
    };
    for (i, layer) in m.layers.iter().enumerate() {
        let p = format!("language_model.model.layers.{i}");
        for (name, l) in [
            ("gate_proj", &layer.moe.shared.gate),
            ("up_proj", &layer.moe.shared.up),
            ("down_proj", &layer.moe.shared.down),
        ] {
            register(l, format!("{p}.mlp.shared_expert.{name}"));
        }
        match &layer.attention {
            HybridAttention::Linear(g) => {
                for (name, l) in [
                    ("in_proj_qkv", &g.qkv),
                    ("in_proj_z", &g.z),
                    ("out_proj", &g.out),
                ] {
                    register(l, format!("{p}.linear_attn.{name}"));
                }
            }
            HybridAttention::Full(g) => {
                for (name, l) in [
                    ("q_proj", &g.attention.q),
                    ("k_proj", &g.attention.k),
                    ("v_proj", &g.attention.v),
                    ("o_proj", &g.attention.o),
                ] {
                    register(l, format!("{p}.self_attn.{name}"));
                }
            }
        }
    }
    ensure!(!names.is_empty(), "no resident calibration projections");
    NAMES.with(|n| *n.borrow_mut() = names);
    STATS.with(|s| s.borrow_mut().clear());
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(false));
            NAMES.with(|n| n.borrow_mut().clear());
            STATS.with(|s| s.borrow_mut().clear());
        }
    }
    let reset = Reset;
    ACTIVE.with(|v| v.set(true));
    run(m)?;
    let stats = STATS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    drop(reset);
    ensure!(!stats.is_empty(), "no activation moments captured");
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{Quantization, Weights};
    use mlx_rs::{Array, ops};
    #[test]
    fn capture_counts_once_and_cleans_registry_after_success_or_error() -> Result<()> {
        let path = std::path::Path::new("tests/fixtures/hybrid");
        let w = Weights::load(path)?;
        let mut m = HybridModel::load(&w, path)?;
        let k = m.config.hidden_size;
        let source = Array::from_iter(
            (0..1024 * k).map(|i| (i as f32 * 0.012).sin() * 0.1),
            &[1024, k],
        )
        .as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = ops::quantize(source, 32, 8)?;
        m.layers[0].moe.shared.gate = Linear {
            weight,
            scales: Some(scales),
            biases: Some(biases),
            bias: None,
            quant: Some(Quantization {
                bits: 8,
                group_size: 32,
                mode: "affine".into(),
            }),
        };
        let data = (0..3 * k).map(|i| (i % 11 - 5) as f32).collect::<Vec<_>>();
        let x = Array::from_slice(&data, &[1, 3, k]).as_dtype(Dtype::Bfloat16)?;
        let name = "language_model.model.layers.0.mlp.shared_expert.gate_proj";
        let stats = capture(&m, |m| {
            m.layers[0].moe.shared.gate.forward_rows(&x)?.eval()?;
            Ok(())
        })?;
        assert!(!active());
        assert_eq!(stats[name].rows, 3);
        for col in 0..k as usize {
            let expected = (0..3)
                .map(|row| data[row * k as usize + col].powi(2) as f64)
                .sum::<f64>();
            assert_eq!(stats[name].sum_squares[col], expected);
        }
        assert!(
            capture(&m, |m| {
                m.layers[0].moe.shared.gate.forward_rows(&x)?.eval()?;
                anyhow::bail!("intentional capture failure")
            })
            .is_err()
        );
        assert!(!active());
        let stats = capture(&m, |m| {
            m.layers[0].moe.shared.gate.forward_rows(&x)?.eval()?;
            Ok(())
        })?;
        assert_eq!(stats[name].rows, 3);
        Ok(())
    }
}
