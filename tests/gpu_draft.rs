//! Lazy GPU token chains must preserve all private MTP transitions.
use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype, ops,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{hybrid::HybridConfig, mtp::Mtp, qsa::QsaCache, weights::Weights};

fn exact(a: &Array, b: &Array) -> Result<()> {
    ensure!(
        a.shape() == b.shape() && a.dtype() == b.dtype(),
        "layout differs"
    );
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    ensure!(a.as_slice::<f32>() == b.as_slice::<f32>(), "values differ");
    Ok(())
}

fn cache_exact(a: &QsaCache, b: &QsaCache) -> Result<()> {
    ensure!(
        a.kv.offset == b.kv.offset && a.kv.rope_offset == b.kv.rope_offset,
        "positions differ"
    );
    for (a, b) in [
        (&a.kv.keys, &b.kv.keys),
        (&a.kv.values, &b.kv.values),
        (&a.raw_keys, &b.raw_keys),
        (&a.blocks, &b.blocks),
    ] {
        match (a, b) {
            (Some(a), Some(b)) => exact(a, b)?,
            (None, None) => {}
            _ => anyhow::bail!("cache presence differs"),
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires local target and saved private MTP oracle"]
fn gpu_token_chains_preserve_private_logits_hidden_and_caches() -> Result<()> {
    let model = std::env::var("RUST_MLX_TARGET_MODEL")?;
    let w = Weights::load(std::path::Path::new(&model))?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let m = Mtp::load(&w, &c)?;
    let e = w.linear("language_model.model.embed_tokens")?;
    let head = w.linear("language_model.lm_head")?;
    let oracle = Array::load_safetensors("results/mtp-oracle.safetensors")?;
    let mut initial_cache = QsaCache::default();
    let (initial_mixed, initial_hidden) = m.forward(
        &e.embedding(&oracle["input_tokens"])?,
        &oracle["input_hidden"],
        &mut initial_cache,
        0,
    )?;
    let initial_logits = head.forward(&initial_mixed.index((.., -1.., ..)))?;
    initial_logits.eval()?;
    for depth in 1..=7 {
        rust_mlx::kv_blocks::set_enabled(false);
        let mut baseline_cache = initial_cache.clone();
        let mut hidden = initial_hidden.index((.., -1.., ..));
        let mut logits = initial_logits.clone();
        let mut baseline = Vec::new();
        let mut cpu_ids = Vec::new();
        for _ in 0..depth {
            let token = indexing::argmax(&logits, false)?.item_exact::<u32>();
            cpu_ids.push(token);
            let pos = baseline_cache.kv.offset;
            let (mixed, wide) = m.forward(
                &e.embedding(&Array::from_slice(&[token], &[1, 1]))?,
                &hidden,
                &mut baseline_cache,
                pos,
            )?;
            logits = head.forward(&mixed)?;
            logits.eval()?;
            hidden = wide.clone();
            baseline.push((mixed, wide, logits.clone(), baseline_cache.clone()));
        }
        rust_mlx::kv_blocks::set_enabled(std::env::var_os("RUST_MLX_TEST_KV_BLOCKS").is_some());
        let mut gpu_cache = initial_cache.clone();
        let mut hidden = initial_hidden.index((.., -1.., ..));
        let mut logits = initial_logits.clone();
        let mut candidate = Vec::new();
        let mut gpu_ids = Vec::new();
        for _ in 0..depth {
            let token = indexing::argmax(&logits, false)?.reshape(&[1, 1])?;
            let pos = gpu_cache.kv.offset;
            let (mixed, wide) = m.forward(&e.embedding(&token)?, &hidden, &mut gpu_cache, pos)?;
            gpu_ids.push(token);
            logits = head.forward(&mixed)?;
            hidden = wide.clone();
            candidate.push((mixed, wide, logits.clone(), gpu_cache.clone()));
        }
        // Evaluate the entire dependent chain before observing any intermediate stage.
        logits.eval()?;
        let ids = ops::concatenate(&gpu_ids, 1)?.contiguous()?;
        ids.eval()?;
        ensure!(ids.as_slice::<u32>() == cpu_ids, "draft IDs differ");
        for ((x, h, l, c), (gx, gh, gl, gc)) in baseline.iter().zip(&candidate) {
            exact(x, gx)?;
            exact(h, gh)?;
            exact(l, gl)?;
            cache_exact(c, gc)?;
        }
        println!("GPU_DRAFT_CHAIN_EXACT depth={depth}");
        rust_mlx::kv_blocks::set_enabled(false);
    }
    Ok(())
}
