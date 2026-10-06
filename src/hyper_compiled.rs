//! Experimental pure hyper-connection graph. Every tensor is a compiler input.
use crate::{
    hybrid::HyperConnection,
    weights::{Linear, Quantization},
};
use anyhow::Result;
use mlx_rs::{
    Array,
    error::{Exception, Result as MlxResult},
    transforms::compile::compile,
};
use std::{cell::RefCell, collections::HashMap};
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct LinearMeta {
    quant: Option<(i32, i32)>,
    scales: bool,
    biases: bool,
    bias: bool,
}
impl LinearMeta {
    fn from(l: &Linear) -> Self {
        Self {
            quant: l.quant.as_ref().map(|q| (q.group_size, q.bits)),
            scales: l.scales.is_some(),
            biases: l.biases.is_some(),
            bias: l.bias.is_some(),
        }
    }
    fn read(self, args: &[Array], i: &mut usize) -> Linear {
        fn take(args: &[Array], i: &mut usize) -> Array {
            let a = args[*i].clone();
            *i += 1;
            a
        }
        Linear {
            weight: take(args, i),
            scales: self.scales.then(|| take(args, i)),
            biases: self.biases.then(|| take(args, i)),
            bias: self.bias.then(|| take(args, i)),
            quant: self.quant.map(|(group_size, bits)| Quantization {
                group_size,
                bits,
                mode: "affine".into(),
            }),
        }
    }
}
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Plan {
    verify: bool,
    rows: bool,
    qmv: bool,
    qmv_batch: bool,
    qmv_stream_x: bool,
    hc_projection: bool,
    gemv: bool,
    gemv_narrow: bool,
    down: LinearMeta,
    up: LinearMeta,
    inject: Option<LinearMeta>,
    hc: i32,
    eps: u32,
}
type Graph = Box<dyn FnMut(&[Array]) -> MlxResult<Vec<Array>>>;
thread_local! {static GRAPHS:RefCell<HashMap<Plan,Graph>>=RefCell::new(HashMap::new());}
fn append(l: &Linear, args: &mut Vec<Array>) {
    args.push(l.weight.clone());
    for a in [&l.scales, &l.biases, &l.bias].into_iter().flatten() {
        args.push(a.clone());
    }
}
pub fn forward(h: &HyperConnection, x: &Array) -> Result<(Array, Option<Array>)> {
    let p = Plan {
        verify: crate::verification::active(),
        rows: crate::verification::rows(),
        qmv: crate::qmv_kernel::enabled(),
        qmv_batch: crate::qmv_kernel::batch_enabled(),
        qmv_stream_x: crate::qmv_kernel::stream_x(),
        hc_projection: crate::hc_kernel::enabled(),
        gemv: crate::gemv_kernel::enabled(),
        gemv_narrow: crate::gemv_kernel::narrow(),
        down: LinearMeta::from(&h.down),
        up: LinearMeta::from(&h.up),
        inject: h.inject.as_ref().map(LinearMeta::from),
        hc: h.hc,
        eps: h.eps.to_bits(),
    };
    let mut args = vec![x.clone(), h.scale.clone()];
    append(&h.down, &mut args);
    append(&h.up, &mut args);
    if let Some(i) = &h.inject {
        append(i, &mut args);
    }
    let mut out = GRAPHS.with(|cache| -> MlxResult<Vec<Array>> {
        let mut cache = cache.borrow_mut();
        let f = cache.entry(p).or_insert_with(|| {
            Box::new(compile(
                move |args: &[Array]| -> MlxResult<Vec<Array>> {
                    let mut i = 2;
                    let h = HyperConnection {
                        scale: args[1].clone(),
                        down: p.down.read(args, &mut i),
                        up: p.up.read(args, &mut i),
                        inject: p.inject.map(|m| m.read(args, &mut i)),
                        hc: p.hc,
                        eps: f32::from_bits(p.eps),
                        compiled_mode: std::cell::Cell::new(false),
                    };
                    let (mixed, inject) = h
                        .forward_reference(&args[0])
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    let mut out = vec![mixed];
                    if let Some(inject) = inject {
                        out.push(inject);
                    }
                    Ok(out)
                },
                false,
            ))
        });
        f(&args)
    })?;
    let inject = if out.len() == 2 { out.pop() } else { None };
    Ok((out.remove(0), inject))
}
