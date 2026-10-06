//! Physical-capacity changes must preserve snapshots, accepted prefixes and continuations.
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype, ops};
use rust_mlx::{dense::KvCache, kv_blocks};
fn exact(a: &Array, b: &Array) -> Result<()> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(
        a.shape() == b.shape() && a.as_slice::<f32>() == b.as_slice::<f32>(),
        "cache values differ"
    );
    Ok(())
}
#[test]
fn snapshots_rollback_growth_and_batch_fallback_are_exact() -> Result<()> {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for b in [1, 2] {
            let mut native = KvCache::default();
            let mut candidate = KvCache::default();
            for t in [255, 4, 1, 256, 3] {
                let x = Array::from_iter(
                    (0..b * 2 * t * 128).map(|i| (i as f32 * 0.01).sin()),
                    &[b, 2, t, 128],
                )
                .as_dtype(dtype)?;
                let y = x.multiply(Array::from_f32(0.5))?.as_dtype(dtype)?;
                kv_blocks::set_enabled(false);
                let (nk, nv) = native.update(x.clone(), y.clone())?;
                kv_blocks::set_enabled(true);
                let (ck, cv) = candidate.update(x, y)?;
                exact(&nk, &ck)?;
                exact(&nv, &cv)?;
                let snapshot = candidate.clone();
                let saved = snapshot.keys.as_ref().unwrap().clone();
                let z = ops::zeros_dtype(&[b, 2, 4, 128], dtype)?;
                candidate.update(z.clone(), z.clone())?.0.eval()?;
                exact(snapshot.keys.as_ref().unwrap(), &saved)?;
                candidate.trim(snapshot.offset)?;
                exact(
                    native.keys.as_ref().unwrap(),
                    candidate.keys.as_ref().unwrap(),
                )?;
                candidate = snapshot;
            }
            // Rewind into a retained snapshot, then overwrite its old suffix.
            // Keep a CPU copy: comparing two aliased Array handles could hide mutation.
            let old = candidate
                .keys
                .as_ref()
                .unwrap()
                .as_dtype(Dtype::Float32)?
                .contiguous()?;
            old.eval()?;
            let saved_values = old.as_slice::<f32>().to_vec();
            let mut fork = candidate.clone();
            fork.trim(candidate.offset - 2)?;
            let replacement = ops::ones_dtype(&[b, 2, 4, 128], dtype)?;
            kv_blocks::set_enabled(true);
            fork.update(replacement.clone(), replacement)?.0.eval()?;
            let retained = candidate
                .keys
                .as_ref()
                .unwrap()
                .as_dtype(Dtype::Float32)?
                .contiguous()?;
            retained.eval()?;
            ensure!(
                retained.as_slice::<f32>() == saved_values,
                "retained snapshot was mutated"
            );
            // Rejected verifier suffix is overwritten, even across a physical-capacity boundary.
            let base = candidate.clone();
            let z = ops::zeros_dtype(&[b, 2, 4, 128], dtype)?;
            candidate.update(z.clone(), z.clone())?.0.eval()?;
            candidate.trim(base.offset + 1)?;
            let ones = ops::ones_dtype(&[b, 2, 1, 128], dtype)?;
            candidate.update(ones.clone(), ones.clone())?.0.eval()?;
            let mut expected = base.clone();
            kv_blocks::set_enabled(false);
            expected.update(
                ops::zeros_dtype(&[b, 2, 1, 128], dtype)?,
                ops::zeros_dtype(&[b, 2, 1, 128], dtype)?,
            )?;
            expected.update(ones.clone(), ones)?.0.eval()?;
            exact(
                expected.keys.as_ref().unwrap(),
                candidate.keys.as_ref().unwrap(),
            )?;
            exact(
                expected.values.as_ref().unwrap(),
                candidate.values.as_ref().unwrap(),
            )?;
        }
    }
    kv_blocks::set_enabled(false);
    Ok(())
}
