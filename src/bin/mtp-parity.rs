use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{hybrid::HybridConfig, mtp::Mtp, qsa::QsaCache, weights::Weights};
fn exact(name: &str, x: &Array, o: &std::collections::HashMap<String, Array>) -> Result<()> {
    let x = x.as_dtype(Dtype::Float32)?.contiguous()?;
    let y = o[name].as_dtype(Dtype::Float32)?.contiguous()?;
    mlx_rs::transforms::eval([&x, &y])?;
    let max = x
        .as_slice::<f32>()
        .iter()
        .zip(y.as_slice::<f32>())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("{name}: max_error={max}");
    ensure!(max == 0., "MTP stage differs");
    Ok(())
}
fn main() -> Result<()> {
    let path =
        std::path::Path::new("/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp");
    let w = Weights::load(path)?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let m = Mtp::load(&w, &c)?;
    let o = Array::load_safetensors("results/mtp-oracle.safetensors")?;
    let e = w.linear("language_model.model.embed_tokens")?;
    let head = w.linear("language_model.lm_head")?;
    let mut cache = QsaCache::default();
    let (mixed, mut wide) = m.forward(
        &e.embedding(&o["input_tokens"])?,
        &o["input_hidden"],
        &mut cache,
        0,
    )?;
    exact("prefill_mixed", &mixed, &o)?;
    exact("prefill_hidden", &wide, &o)?;
    let mut logits = head.forward(&mixed.index((.., -1.., ..)))?;
    exact("prefill_logits", &logits, &o)?;
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read("results/mtp-oracle.json")?)?;
    for i in 0..3 {
        let token = indexing::argmax(&logits, false)?.item_exact::<u32>();
        ensure!(
            token == expected["tokens"][i].as_u64().unwrap() as u32,
            "MTP token mismatch"
        );
        if i < 2 {
            let emb = e.embedding(&Array::from_slice(&[token], &[1, 1]))?;
            let (mixed, h) =
                m.forward(&emb, &wide.index((.., -1.., ..)), &mut cache, 10 + i as i32)?;
            wide = h;
            logits = head.forward(&mixed)?;
            exact(&format!("decode_{i}_mixed"), &mixed, &o)?;
            exact(&format!("decode_{i}_hidden"), &wide, &o)?;
            exact(&format!("decode_{i}_logits"), &logits, &o)?;
        }
    }
    println!("MTP_PARITY_PASSED");
    Ok(())
}
