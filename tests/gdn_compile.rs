//! Compare complete model transitions, including weights shared by structural plans.
use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{
    gdn_compiled,
    hybrid::{HybridCache, HybridModel, LayerCache},
    verification,
    weights::Weights,
};
use std::path::Path;

fn exact(a: &Array, b: &Array, label: &str) -> Result<()> {
    ensure!(a.shape() == b.shape(), "{label}: shape mismatch");
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    let b = b.as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&a, &b])?;
    let mut max = 0f32;
    for (&x, &y) in a.as_slice::<f32>().iter().zip(b.as_slice::<f32>()) {
        ensure!(x.is_finite() && y.is_finite(), "{label}: non-finite result");
        max = max.max((x - y).abs());
    }
    ensure!(max == 0., "{label}: max error {max}");
    Ok(())
}
fn optional(a: &Option<Array>, b: &Option<Array>, label: &str) -> Result<()> {
    match (a, b) {
        (Some(a), Some(b)) => exact(a, b, label),
        (None, None) => Ok(()),
        _ => anyhow::bail!("{label}: initialization differs"),
    }
}
fn caches(a: &HybridCache, b: &HybridCache) -> Result<()> {
    ensure!(
        a.offset == b.offset && a.history == b.history,
        "CPU cache mismatch"
    );
    for (i, (a, b)) in a.layers.iter().zip(&b.layers).enumerate() {
        match (a, b) {
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                for (x, y, name) in [
                    (&a.state, &b.state, "state"),
                    (&a.conv, &b.conv, "conv"),
                    (&a.verified_states, &b.verified_states, "history"),
                    (&a.verified_conv, &b.verified_conv, "conv history"),
                ] {
                    optional(x, y, &format!("layer{i} GDN {name}"))?;
                }
            }
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                ensure!(
                    a.kv.offset == b.kv.offset && a.kv.rope_offset == b.kv.rope_offset,
                    "KV position mismatch"
                );
                for (x, y, name) in [
                    (&a.kv.keys, &b.kv.keys, "keys"),
                    (&a.kv.values, &b.kv.values, "values"),
                    (&a.raw_keys, &b.raw_keys, "raw index keys"),
                    (&a.blocks, &b.blocks, "summaries"),
                ] {
                    optional(x, y, &format!("layer{i} QSA {name}"))?;
                }
            }
            _ => anyhow::bail!("layer{i}: cache type mismatch"),
        }
    }
    for (i, (a, b)) in a.ple.iter().zip(&b.ple).enumerate() {
        optional(&a.conv, &b.conv, &format!("layer{i} PLE"))?;
    }
    Ok(())
}
fn compare(model: &Path, prompt: &[u32], require_compilation: bool) -> Result<()> {
    mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
    mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
    let w = Weights::load(model)?;
    let m = HybridModel::load(&w, model)?;
    gdn_compiled::set_enabled(false);
    let mut reference = m.make_cache();
    let (mut logits, _) = m.forward(prompt, &mut reference)?;
    logits.eval()?;
    let mut candidate = reference.clone();
    let start = gdn_compiled::calls();
    for step in 0..16 {
        let token = indexing::argmax(logits.index((0, -1, ..)), false)?.item_exact::<u32>();
        gdn_compiled::set_enabled(false);
        let (l, h) = m.forward(&[token], &mut reference)?;
        gdn_compiled::set_enabled(true);
        let (cl, ch) = m.forward(&[token], &mut candidate)?;
        exact(&l, &cl, &format!("plain step{step} logits"))?;
        exact(&h, &ch, &format!("plain step{step} hidden"))?;
        caches(&reference, &candidate)?;
        logits = l;
        println!("COMPILED_GDN_PLAIN_STEP_EXACT {step}");
    }
    for depth in [2, 4, 8] {
        let token = indexing::argmax(logits.index((0, -1, ..)), false)?.item_exact::<u32>();
        let tokens = vec![token; depth];
        let mut rc = reference.clone();
        let mut cc = reference.clone();
        gdn_compiled::set_enabled(false);
        let (l, h) = verification::with_mode(|| m.forward(&tokens, &mut rc))?;
        gdn_compiled::set_enabled(true);
        let (cl, ch) = verification::with_mode(|| m.forward(&tokens, &mut cc))?;
        exact(&l, &cl, &format!("verify depth{depth} logits"))?;
        exact(&h, &ch, &format!("verify depth{depth} hidden"))?;
        caches(&rc, &cc)?;
        println!("COMPILED_GDN_VERIFY_EXACT {depth}");
    }
    ensure!(
        (gdn_compiled::calls() > start) == require_compilation,
        "unexpected compiled graph engagement"
    );
    gdn_compiled::set_enabled(false);
    Ok(())
}
#[test]
fn f32_fallback_preserves_fixture_logits_and_complete_caches() -> Result<()> {
    compare(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hybrid"),
        &[2, 17, 43, 85],
        false,
    )
}
#[test]
#[ignore = "requires local target checkpoint in RUST_MLX_TARGET_MODEL"]
fn pure_gdn_graph_preserves_target_logits_and_complete_caches() -> Result<()> {
    let model = std::env::var("RUST_MLX_TARGET_MODEL")?;
    compare(
        Path::new(&model),
        &[7734, 264, 2716, 32671, 709, 421, 55288, 76938, 4947, 13],
        true,
    )
}

#[test]
fn bf16_mixed_scale_graph_preserves_dynamic_weights_and_state() -> Result<()> {
    use rust_mlx::{
        hybrid::{Gdn, GdnCache, HybridConfig},
        weights::Linear,
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hybrid");
    let w = Weights::load(&path)?;
    let config: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let cast = |l: &mut Linear| -> Result<()> {
        if l.weight.dtype() == Dtype::Float32 {
            l.weight = l.weight.as_dtype(Dtype::Bfloat16)?;
        }
        for value in [&mut l.scales, &mut l.biases, &mut l.bias]
            .into_iter()
            .flatten()
        {
            *value = value.as_dtype(Dtype::Bfloat16)?;
        }
        Ok(())
    };
    // Reuse the same structural graph with three different sets of weights.
    for layer in 0..3 {
        let mut g = Gdn::load(
            &w,
            &format!("language_model.model.layers.{layer}.linear_attn"),
            &config,
        )?;
        for l in [&mut g.qkv, &mut g.z, &mut g.out] {
            cast(l)?;
        }
        // Actual target a/b projections deliberately retain Float32 scales.
        g.conv = rust_mlx::conv_weights::guarded(&g.conv.as_dtype(Dtype::Bfloat16)?)?;
        g.decode_conv = g.conv.index((.., .., 0)).t().as_dtype(Dtype::Float32)?;
        g.dt = g.dt.as_dtype(Dtype::Bfloat16)?;
        g.norm = g.norm.as_dtype(Dtype::Bfloat16)?;
        let input = |batch, time, seed: i32| -> Result<Array> {
            Ok(Array::from_iter(
                (0..batch * time * config.hidden_size).map(|i| ((i + seed) as f32 * 0.017).sin()),
                &[batch, time, config.hidden_size],
            )
            .as_dtype(Dtype::Bfloat16)?)
        };
        for (batch, time, verify) in [
            (1, 1, false),
            (2, 1, false),
            (4, 1, false),
            (8, 1, false),
            (1, 2, true),
            (1, 4, true),
            (1, 8, true),
        ] {
            let mut base = GdnCache::default();
            gdn_compiled::set_enabled(false);
            g.forward(&input(batch, 3, layer)?, &mut base)?.eval()?;
            let mut candidate = base.clone();
            for step in 0..3 {
                let x = input(batch, time, step + layer * 19)?;
                let run = |cache: &mut GdnCache| {
                    if verify {
                        verification::with_mode(|| g.forward(&x, cache))
                    } else if batch > 1 {
                        verification::with_rows(|| g.forward(&x, cache))
                    } else {
                        g.forward(&x, cache)
                    }
                };
                gdn_compiled::set_enabled(false);
                let y = run(&mut base)?;
                let before = gdn_compiled::calls();
                gdn_compiled::set_enabled(true);
                let compiled = run(&mut candidate)?;
                ensure!(
                    gdn_compiled::calls() > before,
                    "BF16 compiled graph did not engage"
                );
                exact(&y, &compiled, "BF16 recurrent output")?;
                for (a, b, label) in [
                    (&base.state, &candidate.state, "BF16 state"),
                    (&base.conv, &candidate.conv, "BF16 convolution"),
                    (
                        &base.verified_states,
                        &candidate.verified_states,
                        "BF16 state history",
                    ),
                    (
                        &base.verified_conv,
                        &candidate.verified_conv,
                        "BF16 convolution history",
                    ),
                ] {
                    optional(a, b, label)?;
                }
            }
        }
    }
    gdn_compiled::set_enabled(false);
    Ok(())
}
