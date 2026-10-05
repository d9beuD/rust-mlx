//! Oracle-compatible indexed QSA prefill. Singleton decode retains native SDPA.
use crate::metal::{Kernel, Launch, Template};
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype, ops};
use std::cell::RefCell;
thread_local! {static KERNEL:RefCell<Option<Kernel>>=const{RefCell::new(None)};}
pub fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    blocks: &Array,
    ends: &Array,
    ratio: i32,
    scale: f32,
) -> Result<Array> {
    ensure!(
        q.ndim() == 4
            && k.ndim() == 4
            && k.shape() == v.shape()
            && blocks.ndim() == 3
            && ends.ndim() == 2,
        "invalid QSA dimensions"
    );
    let (b, h, t, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let kh = k.shape()[1];
    let topk = blocks.shape()[2];
    ensure!(
        b > 0
            && h > 0
            && t > 0
            && d > 0
            && kh > 0
            && topk > 0
            && d % 32 == 0
            && k.shape()[0] == b
            && k.shape()[3] == d
            && h % kh == 0
            && blocks.shape() == [b, t, topk]
            && ends.shape() == [b, t]
            && ratio > 0,
        "invalid QSA indexing"
    );
    ensure!(
        matches!(q.dtype(), Dtype::Bfloat16 | Dtype::Float16)
            && q.dtype() == k.dtype()
            && q.dtype() == v.dtype(),
        "unsupported QSA dtype"
    );
    let blocks = ops::sort_axis(blocks.as_dtype(Dtype::Int32)?, -1)?.contiguous()?;
    let s = Array::from_slice(&[scale], &[1]);
    let len = Array::from_slice(&[k.shape()[2]], &[1]);
    KERNEL.with(|cell| {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::new(
                "rust_mlx_qsa_prefill",
                &[
                    "queries",
                    "keys",
                    "values",
                    "block_indices",
                    "query_ends",
                    "scale",
                    "k_size",
                ],
                &["out"],
                include_str!("../kernels/qsa_attention.metal"),
            )?);
        }
        Ok(cell
            .as_ref()
            .expect("initialized above")
            .launch(Launch {
                inputs: &[q, k, v, &blocks, ends, &s, &len],
                templates: &[
                    Template::Dtype("T", q.dtype()),
                    Template::Int("D_SIZE", d),
                    Template::Int("Q_LEN", t),
                    Template::Int("NUM_Q_HEADS", h),
                    Template::Int("NUM_KV_HEADS", kh),
                    Template::Int("GQA_FACTOR", h / kh),
                    Template::Int("BLOCK_SIZE", ratio),
                    Template::Int("TOPK_BLOCKS", topk),
                    Template::Int("SELECTED_LENGTH", topk * ratio),
                ],
                outputs: &[(q.shape(), q.dtype())],
                grid: [1024, b * h * t, 1],
                group: [1024, 1, 1],
            })?
            .remove(0))
    })
}
