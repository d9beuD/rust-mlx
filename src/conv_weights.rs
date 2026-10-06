//! Backing allocation guard for pinned MLX 0.32.2 depthwise convolutions.
use mlx_rs::{Array, error::Result, ops, ops::indexing::IndexOp};

/// Preserve the logical weights while giving MLX's BN8/BK16 loader a valid tail.
/// Its grouped channel-one loader checks tail output lanes against total O,
/// instead of per-group N. Zero activation lanes discard those extra reads,
/// but the reads must still remain within an owned Metal buffer. No unsafe
/// pointer or shape manipulation is needed: the evaluated prefix is a view
/// into the larger contiguous allocation, with unchanged values and strides.
pub fn guarded(weight: &Array) -> Result<Array> {
    if weight.ndim() != 3 || weight.shape()[2] != 1 {
        return Ok(weight.clone());
    }
    let channels = weight.shape()[0];
    let tail = ops::zeros_dtype(&[32, weight.shape()[1], 1], weight.dtype())?;
    let allocation = ops::concatenate(&[weight, &tail], 0)?;
    allocation.eval()?;
    let view = allocation.index((..channels, .., ..));
    view.eval()?;
    Ok(view)
}
