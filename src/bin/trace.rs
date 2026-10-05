use anyhow::Result;
use mlx_rs::{Array, Dtype};
use rust_mlx::weights::Weights;
fn compare(name: &str, x: &Array, oracle: &std::collections::HashMap<String, Array>) -> Result<()> {
    let y = x.as_dtype(Dtype::Float32)?.contiguous()?;
    let r = oracle[name].as_dtype(Dtype::Float32)?.contiguous()?;
    y.eval()?;
    r.eval()?;
    let ys = y.as_slice::<f32>();
    let rs = r.as_slice::<f32>();
    let max = ys
        .iter()
        .zip(rs)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let different = ys.iter().zip(rs).filter(|(x, y)| x != y).count();
    eprintln!(
        "{name}: {:?} {:?} max_error={max} different={different}/{}",
        x.shape(),
        x.dtype(),
        ys.len()
    );
    Ok(())
}
fn main() -> Result<()> {
    use rust_mlx::hybrid::{HybridAttention, HybridModel, LayerCache};
    let path =
        std::path::Path::new("/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp");
    let w = Weights::load(path)?;
    let m = HybridModel::load(&w, path)?;
    let o = Array::load_safetensors("results/target-trace-oracle.safetensors")?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read("results/target-trace-prompt.json")?)?;
    let mut cache = m.make_cache();
    let h = m
        .embedding
        .embedding(&Array::from_slice(&tokens, &[1, tokens.len() as i32]))?;
    let h = mlx_rs::ops::broadcast_to(&h.expand_dims(2)?, &[1, tokens.len() as i32, 4, 2560])?
        .reshape(&[1, tokens.len() as i32, 10240])?
        .contiguous()?;
    compare("embedding", &h, &o)?;
    for (i, l) in m.layers.iter().enumerate() {
        let mut h = o[&format!("{i}.input")].clone();
        if let Some(p) = &m.ple[i] {
            h = p.forward(&h, &tokens, &[], &mut cache.ple[i])?;
        }
        compare(&format!("{i}.after_ple"), &h, &o)?;
        let (mixed, inj) = l.attn_hc.forward(&o[&format!("{i}.after_ple")])?;
        compare(&format!("{i}.mixed"), &mixed, &o)?;
        let branch = match (&l.attention, &mut cache.layers[i]) {
            (HybridAttention::Linear(a), LayerCache::Linear(c)) => {
                a.forward(&o[&format!("{i}.mixed")], c)?
            }
            (HybridAttention::Full(a), LayerCache::Full(c)) => {
                a.forward(&o[&format!("{i}.mixed")], c)?
            }
            _ => anyhow::bail!("cache mismatch"),
        };
        if i == 3
            && let HybridAttention::Full(a) = &l.attention
        {
            use mlx_rs::ops::indexing::IndexOp;
            let x = &o["3.mixed"];
            let att = &a.attention;
            let qr = att.q.forward(x)?;
            let kr = att.k.forward(x)?;
            let vr = att.v.forward(x)?;
            compare("qraw", &qr, &o)?;
            compare("kraw", &kr, &o)?;
            compare("vraw", &vr, &o)?;
            let qg = qr.reshape(&[1, 10, 24, 2, 256])?;
            let qq = qg.index((.., .., .., 0, ..));
            let gg = qg.index((.., .., .., 1, ..));
            let qq = rust_mlx::compiled::norm(&qq, &att.qnorm, att.eps)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let kk = rust_mlx::compiled::norm(&kr.reshape(&[1, 10, 2, 256])?, &att.knorm, att.eps)?
                .transpose_axes(&[0, 2, 1, 3])?;
            compare("qnorm", &qq, &o)?;
            compare("knorm", &kk, &o)?;
            let qq = rust_mlx::rope::text(&qq, 64, att.theta, 0, 1)?;
            let kk = rust_mlx::rope::text(&kk, 64, att.theta, 0, 1)?;
            compare("qrope", &qq, &o)?;
            compare("krope", &kk, &o)?;
            let vv = vr
                .reshape(&[1, 10, 2, 256])?
                .transpose_axes(&[0, 2, 1, 3])?;
            compare("values", &vv, &o)?;
            let oo = mlx_rs::fast::scaled_dot_product_attention(
                &qq,
                &kk,
                &vv,
                256f32.powf(-0.5),
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal),
                None,
            )?;
            compare("sdpa", &oo, &o)?;
            let gg = gg.reshape(&[1, 10, 6144])?;
            compare("gate", &gg, &o)?;
            let gated = oo
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, 10, 6144])?
                .multiply(mlx_rs::ops::sigmoid(&gg)?)?;
            compare("gated", &gated, &o)?;
            compare("3.branch", &att.o.forward(&gated)?, &o)?;
            compare(
                "3.branch",
                &att.forward(x, &mut rust_mlx::dense::KvCache::default())?,
                &o,
            )?;
            eprintln!(
                "att_config theta={} rotary={} eps={}",
                att.theta, att.rotary_dim, att.eps
            );
        }
        compare(&format!("{i}.branch"), &branch, &o)?;
        h = l.attn_hc.write(
            &o[&format!("{i}.after_ple")],
            &o[&format!("{i}.branch")],
            inj.as_ref().unwrap(),
        )?;
        compare(&format!("{i}.after_attention"), &h, &o)?;
        let (mixed, inj) = l.mlp_hc.forward(&o[&format!("{i}.after_attention")])?;
        compare(&format!("{i}.mlp_input"), &mixed, &o)?;
        let branch = l.moe.forward(&o[&format!("{i}.mlp_input")])?;
        compare(&format!("{i}.moe"), &branch, &o)?;
        h = l.mlp_hc.write(
            &o[&format!("{i}.after_attention")],
            &o[&format!("{i}.moe")],
            inj.as_ref().unwrap(),
        )?;
        compare(&format!("{i}.output"), &h, &o)?;
    }
    Ok(())
}
