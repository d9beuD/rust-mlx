//! Exact experimental shared-expert/add/injection/residual epilogue.
use crate::metal::{Kernel, Launch};
use anyhow::Result;
use mlx_rs::{Array, Dtype};
use std::cell::{Cell, RefCell};
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) };
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|v| v.set(value));
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}
pub fn apply(
    routed: &Array,
    shared: &Array,
    factor: &Array,
    residual: &Array,
    gate: &Array,
    hc: i32,
) -> Result<Option<Array>> {
    if !ENABLED.with(Cell::get)
        || hc != 4
        || routed.ndim() != 3
        || routed.shape()[0] != 1
        || !(1..=8).contains(&routed.shape()[1])
        || routed.shape()[2] != 2560
        || shared.shape() != routed.shape()
        || residual.shape() != [1, routed.shape()[1], hc * routed.shape()[2]]
        || gate.shape() != [1, routed.shape()[1], hc]
        || factor.shape() != [1, routed.shape()[1], 1]
        || [routed, shared, factor, residual, gate]
            .iter()
            .any(|x| x.dtype() != Dtype::Bfloat16)
    {
        return Ok(None);
    }
    KERNEL.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Kernel::new(
                "rust_mlx_moe_hc_epilogue",
                &["routed", "shared", "factor", "residual", "gate"],
                &["out"],
                include_str!("../kernels/moe_hc_epilogue.metal"),
            )?);
        }
        let inputs = [routed, shared, factor, residual, gate]
            .map(|x| x.contiguous())
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let references = inputs.iter().collect::<Vec<_>>();
        let rows = routed.shape()[1];
        let hidden = routed.shape()[2];
        let mut output = slot.as_ref().expect("initialized").launch(Launch {
            inputs: &references,
            templates: &[
                crate::metal::Template::Dtype("T", Dtype::Bfloat16),
                crate::metal::Template::Int("ROWS", rows),
                crate::metal::Template::Int("HC", hc),
                crate::metal::Template::Int("HIDDEN", hidden),
            ],
            outputs: &[(residual.shape(), Dtype::Bfloat16)],
            grid: [rows * hc * hidden, 1, 1],
            group: [256, 1, 1],
        })?;
        CALLS.with(|c| c.set(c.get().wrapping_add(1)));
        Ok(output.pop())
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn epilogue_preserves_every_bf16_boundary() {
        set_enabled(true);
        for t in [1, 2, 3, 4, 8] {
            let sample = |n: i32, shape: &[i32], phase: f32| {
                Array::from_iter(
                    (0..n).map(|i| (i as f32 * 0.073 + phase).sin() * 7.3),
                    shape,
                )
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
            };
            let routed = sample(t * 2560, &[1, t, 2560], 0.);
            let shared = sample(t * 2560, &[1, t, 2560], 1.);
            let factor = sample(t, &[1, t, 1], 2.);
            let residual = sample(t * 10240, &[1, t, 10240], 3.);
            let gate = sample(t * 4, &[1, t, 4], 4.);
            let branch = routed.add(shared.multiply(&factor).unwrap()).unwrap();
            let reference = residual
                .reshape(&[1, t, 4, 2560])
                .unwrap()
                .add(
                    branch
                        .expand_dims(2)
                        .unwrap()
                        .multiply(gate.expand_dims(-1).unwrap())
                        .unwrap(),
                )
                .unwrap()
                .reshape(&[1, t, 10240])
                .unwrap();
            let actual = apply(&routed, &shared, &factor, &residual, &gate, 4)
                .unwrap()
                .unwrap();
            let a = actual
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            let b = reference
                .as_dtype(Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap();
            mlx_rs::transforms::eval([&a, &b]).unwrap();
            assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
        }
        set_enabled(false);
    }
}
