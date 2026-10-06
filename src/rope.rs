//! Text-only MRoPE with the oracle's stored float32 inverse frequencies.
use crate::metal::{Kernel, Launch, Template};
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype};
use std::{cell::RefCell, collections::HashMap};
thread_local! { static KERNEL:RefCell<Option<Kernel>>=const{RefCell::new(None)};
static FREQ:RefCell<HashMap<(i32,u32),Array>>=RefCell::new(HashMap::new()); }
pub fn text(x: &Array, dim: i32, theta: f32, offset: i32, stride: i32) -> Result<Array> {
    ensure!(
        x.ndim() == 4 && dim > 0 && dim % 2 == 0 && dim <= x.shape()[3],
        "invalid text RoPE shape"
    );
    if x.size() == 0 {
        return Ok(x.clone());
    }
    let freq = FREQ.with(|f| -> Result<Array> {
        let mut f = f.borrow_mut();
        let key = (dim, theta.to_bits());
        if let std::collections::hash_map::Entry::Vacant(e) = f.entry(key) {
            let power = Array::from_iter((0..dim).step_by(2), &[dim / 2])
                .as_dtype(Dtype::Float32)?
                .divide(Array::from_f32(dim as f32))?;
            let inv = Array::from_f32(1.).divide(Array::from_f32(theta).power(power)?)?;
            inv.eval()?;
            e.insert(inv);
        }
        Ok(f[&key].clone())
    })?;
    let b = x.shape()[0];
    let t = x.shape()[2];
    let ids = crate::runtime_prepare::positions(b, t, offset, stride)?;
    KERNEL.with(|k| {
        let mut k = k.borrow_mut();
        if k.is_none() {
            *k = Some(Kernel::new(
                "rust_mlx_text_mrope",
                &["x", "position_ids", "inv_freq"],
                &["x_out"],
                include_str!("../kernels/text_rope.metal"),
            )?);
        }
        let size = b * x.shape()[1] * t * (dim / 2 + x.shape()[3] - dim);
        Ok(k.as_ref()
            .expect("initialized above")
            .launch(Launch {
                inputs: &[x, &ids, &freq],
                templates: &[
                    Template::Int("ROTARY", dim),
                    Template::Dtype("T", x.dtype()),
                ],
                outputs: &[(x.shape(), x.dtype())],
                grid: [size, 1, 1],
                group: [256, 1, 1],
            })?
            .remove(0))
    })
}
