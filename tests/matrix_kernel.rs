//! Orientation, packed-word boundaries and padded-row safety of TensorOps.
//! These tests do not certify approximate projections as production defaults.
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype, ops, ops::indexing::IndexOp};
use rust_mlx::{
    matrix_kernel::{self, InputMode},
    verification,
    weights::{Linear, Quantization},
};

fn linear(bits: i32, group: i32, dtype: Dtype) -> Result<Linear> {
    let w = Array::from_iter(
        (0..64 * 512).map(|i| (i as f32 * 0.013).sin() * 0.08),
        &[64, 512],
    )
    .as_dtype(dtype)?;
    let (weight, scales, biases) = ops::quantize(w, group, bits)?;
    Ok(Linear {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        bias: Some(Array::from_iter((0..64).map(|i| i as f32 / 128.), &[64]).as_dtype(dtype)?),
        quant: Some(Quantization {
            bits,
            group_size: group,
            mode: "affine".into(),
        }),
    })
}
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.shape() == b.shape(), "matrix shape differs");
    let first = a
        .as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .enumerate()
        .find(|(_, (a, b))| a != b);
    ensure!(first.is_none(), "matrix projection differs at {first:?}");
    Ok(())
}

#[test]
fn packed_boundaries_and_padding_match_native_one_hot() -> Result<()> {
    if !matrix_kernel::supported() {
        return Ok(());
    }
    for dtype in [Dtype::Bfloat16, Dtype::Float16] {
        for (bits, group) in [(4, 32), (5, 64), (6, 128), (8, 64)] {
            let l = linear(bits, group, dtype)?;
            // Last packed word, metadata boundary, Q5/Q6 cross-word fields.
            let x = Array::from_iter(
                (0..4 * 512).map(|i| {
                    if i % 512 == [6, 31, 127, 511][i / 512] {
                        1.
                    } else {
                        0.
                    }
                }),
                &[1, 4, 512],
            )
            .as_dtype(dtype)?;
            for t in [1, 2, 3, 4] {
                let x = x.index((.., ..t, ..)).contiguous()?;
                let r = ops::concatenate(
                    &(0..t)
                        .map(|i| l.forward(&x.index((.., i..i + 1, ..))))
                        .collect::<Result<Vec<_>>>()?,
                    1,
                )?;
                for (mode, split) in [
                    (InputMode::Staged, 4),
                    (InputMode::Registers, 4),
                    (InputMode::CompactRegisters, 4),
                    (InputMode::AffineRegisters, 32),
                    (InputMode::AffineCompact, 8),
                    (InputMode::Packed, 8),
                ] {
                    if matrix_kernel::instrumented()
                        && matches!(
                            mode,
                            InputMode::Registers
                                | InputMode::CompactRegisters
                                | InputMode::AffineRegisters
                                | InputMode::AffineCompact
                        )
                    {
                        continue;
                    }
                    if mode == InputMode::Packed && bits != 4 {
                        continue;
                    }
                    let y = matrix_kernel::project(&l, &x, mode, split)?
                        .context("supported projection rejected")?;
                    exact(&r, &y).with_context(|| {
                        format!("one-hot {dtype:?} Q{bits}/g{group} T{t} {mode:?}")
                    })?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn register_inputs_match_staged_and_preserve_zero_bias() -> Result<()> {
    if !matrix_kernel::supported() {
        return Ok(());
    }
    for bits in [4, 5, 6, 8] {
        let l = linear(bits, 64, Dtype::Bfloat16)?;
        let x = Array::from_iter((0..4 * 512).map(|i| (i as f32 * 0.017).cos()), &[1, 4, 512])
            .as_dtype(Dtype::Bfloat16)?;
        for split in [1, 2, 4, 8] {
            if matrix_kernel::instrumented() {
                continue;
            }
            let s = matrix_kernel::project(&l, &x, InputMode::Staged, split)?
                .context("staged unsupported")?;
            let r = matrix_kernel::project(&l, &x, InputMode::Registers, split)?
                .context("register unsupported")?;
            exact(&s, &r).with_context(|| format!("staged/register Q{bits} split{split}"))?;
        }
        let z = ops::zeros_dtype(&[1, 4, 512], Dtype::Bfloat16)?;
        let r = l.forward_rows(&z)?;
        for (mode, split) in [
            (InputMode::Staged, 4),
            (InputMode::Registers, 4),
            (InputMode::CompactRegisters, 4),
            (InputMode::AffineRegisters, 32),
            (InputMode::AffineCompact, 8),
            (InputMode::Packed, 8),
        ] {
            if matrix_kernel::instrumented()
                && matches!(
                    mode,
                    InputMode::Registers
                        | InputMode::CompactRegisters
                        | InputMode::AffineRegisters
                        | InputMode::AffineCompact
                )
            {
                continue;
            }
            if mode == InputMode::Packed && bits != 4 {
                continue;
            }
            exact(
                &r,
                &matrix_kernel::project(&l, &z, mode, split)?.context("zero unsupported")?,
            )
            .with_context(|| format!("zero Q{bits} {mode:?}"))?;
        }
    }
    Ok(())
}

#[test]
fn cooperative_validator_counterexample_is_rejected_with_native_fallback() -> Result<()> {
    if !matrix_kernel::supported() || !matrix_kernel::instrumented() {
        return Ok(());
    }
    // Reproduce, rather than hide, the observed native/register discrepancy.
    let small = linear(4, 32, Dtype::Bfloat16)?;
    let x = Array::from_iter((0..512).map(|i| if i == 6 { 1. } else { 0. }), &[1, 1, 512])
        .as_dtype(Dtype::Bfloat16)?;
    let r = small.forward(&x)?;
    let y = matrix_kernel::project(&small, &x, InputMode::Registers, 4)?
        .context("counterexample unsupported")?;
    ensure!(
        exact(&r, &y).is_err(),
        "cooperative counterexample changed: revisit qualification"
    );
    // A geometry that would otherwise enter selected(), not the small-N guard.
    let mut large = linear(4, 32, Dtype::Bfloat16)?;
    large.weight = ops::tile(&large.weight, &[32, 1])?;
    large.scales = Some(ops::tile(large.scales.as_ref().unwrap(), &[32, 1])?);
    large.biases = Some(ops::tile(large.biases.as_ref().unwrap(), &[32, 1])?);
    large.bias = None;
    let x = ops::tile(&x, &[1, 2, 1])?;
    let native = verification::with_rows(|| large.forward(&x))?;
    matrix_kernel::set_packed(false);
    matrix_kernel::set_enabled(true);
    let count = matrix_kernel::calls();
    let fallback = verification::with_rows(|| large.forward(&x));
    matrix_kernel::set_enabled(false);
    ensure!(
        count == matrix_kernel::calls(),
        "unqualified cooperative path selected"
    );
    exact(&native, &fallback?)?;
    Ok(())
}

#[test]
fn unsupported_geometry_has_native_fallback_and_default_is_off() -> Result<()> {
    let l = linear(4, 64, Dtype::Bfloat16)?;
    ensure!(!matrix_kernel::enabled(), "research path became default");
    for (shape, dtype) in [
        ([1, 4, 512], Dtype::Float32),
        ([2, 2, 512], Dtype::Bfloat16),
        ([1, 5, 512], Dtype::Bfloat16),
        ([1, 8, 512], Dtype::Bfloat16),
        ([1, 2, 513], Dtype::Bfloat16),
    ] {
        let x = ops::zeros_dtype(&shape, dtype)?;
        ensure!(
            matrix_kernel::project(&l, &x, InputMode::AffineCompact, 8)?.is_none(),
            "unsupported matrix shape accepted"
        );
    }
    let x = ops::zeros_dtype(&[1, 2, 512], Dtype::Bfloat16)?;
    let r = verification::with_rows(|| l.forward(&x))?;
    matrix_kernel::set_enabled(true);
    // Small output geometry retains the native path even when requested.
    let candidate = verification::with_rows(|| l.forward(&x));
    matrix_kernel::set_enabled(false);
    exact(&r, &candidate?)?;
    Ok(())
}
