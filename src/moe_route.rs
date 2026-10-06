//! Experimental MTPLX routing tail; native affine projections and fallback.
use crate::metal::{Kernel, Launch};
use anyhow::Result;
use mlx_rs::{Array, Dtype};
use std::cell::{Cell, RefCell};

thread_local! {
    static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) };
    static ENABLED: Cell<bool> = Cell::new(std::env::var("RUST_MLX_ROUTE_TAIL").is_ok_and(|v|v=="1"));
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

pub fn set_enabled(enabled: bool) {
    ENABLED.with(|v| v.set(enabled));
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}

/// The checkpoint's finite BF16 logits,512 experts, top10,1–8 rows.
/// Unsupported geometries decline before building a custom graph.
pub fn tail(logits: &Array, shared: &Array, top_k: i32) -> Result<Option<(Array, Array, Array)>> {
    if !enabled()
        || top_k != 10
        || logits.ndim() != 3
        || logits.shape()[0] != 1
        || !(1..=8).contains(&logits.shape()[1])
        || logits.shape()[2] != 512
        || logits.dtype() != Dtype::Bfloat16
        || shared.shape() != [1, logits.shape()[1], 1]
        || shared.dtype() != Dtype::Bfloat16
    {
        return Ok(None);
    }
    KERNEL.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            eprintln!("Powered by MTPLX — https://github.com/youssofal/mtplx");
            *slot = Some(Kernel::with_header(
                "rust_mlx_mtplx_route_tail",
                &["logits", "shared_logits"],
                &["expert_ids", "route_scores", "shared_factor"],
                include_str!("../kernels/moe_route_tail.metal"),
                include_str!("../kernels/moe_route_tail.h"),
            )?);
        }
        let rows = logits.shape()[1];
        let logits = logits.contiguous()?;
        let shared = shared.contiguous()?;
        let mut out = slot.as_ref().expect("initialized above").launch(Launch {
            inputs: &[&logits, &shared],
            templates: &[],
            outputs: &[
                (&[1, rows, 10], Dtype::Uint32),
                (&[1, rows, 10], Dtype::Bfloat16),
                (&[1, rows, 1], Dtype::Bfloat16),
            ],
            grid: [128 * rows, 1, 1],
            group: [128, 1, 1],
        })?;
        CALLS.with(|v| v.set(v.get().wrapping_add(1)));
        let factor = out.pop().expect("three outputs");
        let scores = out.pop().expect("three outputs");
        let ids = out.pop().expect("three outputs");
        Ok(Some((ids, scores, factor)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{self, indexing::IndexOp};

    fn equal(a: &Array, b: &Array) {
        let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
    }

    #[test]
    fn routing_tail_matches_native_scores_ties_and_shared_sigmoid() {
        set_enabled(true);
        for rows in [1, 2, 3, 4, 8] {
            for phase in [0, 1, 2] {
                let logits = Array::from_iter(
                    (0..rows * 512).map(|i| match phase {
                        0 => 0.,
                        1 => ((i * 13 % 512) as f32 - 255.) / 17.,
                        _ => (i as f32 * 0.071).sin() * 12.,
                    }),
                    &[1, rows, 512],
                )
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
                let shared =
                    Array::from_iter((0..rows).map(|i| i as f32 * 2.3 - 7.), &[1, rows, 1])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap();
                let gates = ops::softmax_axis(&logits, -1, true).unwrap();
                let ids = ops::argpartition_axis(&gates, -10, -1)
                    .unwrap()
                    .index((.., .., -10..));
                let scores = gates.take_along_axis(&ids, -1).unwrap();
                let scores = scores.divide(scores.sum_axis(-1, true).unwrap()).unwrap();
                let (actual_ids, actual_scores, factor) =
                    tail(&logits, &shared, 10).unwrap().unwrap();
                equal(&actual_ids, &ids);
                equal(&actual_scores, &scores);
                equal(&factor, &ops::sigmoid(&shared).unwrap());
            }
        }
        set_enabled(false);
        assert!(
            tail(&Array::from_f32(0.), &Array::from_f32(0.), 10)
                .unwrap()
                .is_none()
        );
    }
}
