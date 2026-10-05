use anyhow::Result;
use mlx_rs::{Array, Dtype};
use rust_mlx::{
    hybrid::{Gdn, GdnCache, HybridConfig, HyperConnection, MoE},
    weights::Weights,
};
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
    let p =
        std::path::Path::new("/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp");
    let w = Weights::load(p)?;
    let c: HybridConfig = serde_json::from_value(w.config["text_config"].clone())?;
    let o = Array::load_safetensors("results/target-layer0-oracle.safetensors")?;
    let x = &o["input"];
    let p = "language_model.model.layers.0";
    let ah = HyperConnection::load(&w, &format!("{p}.attn_hyper_connection"), &c, true)?;
    let mh = HyperConnection::load(&w, &format!("{p}.mlp_hyper_connection"), &c, true)?;
    let g = Gdn::load(&w, &format!("{p}.linear_attn"), &c)?;
    let m = MoE::load(&w, &format!("{p}.mlp"), &c)?;
    let (mixed, inject) = ah.forward(x)?;
    compare("mixed", &mixed, &o)?;
    compare("inject", inject.as_ref().unwrap(), &o)?;
    let raw = g.qkv.forward(&mixed)?;
    compare("raw", &raw, &o)?;
    let inp = mlx_rs::ops::concatenate(
        &[
            &mlx_rs::ops::zeros_dtype(&[1, 3, 10240], mixed.dtype())?,
            &raw,
        ],
        1,
    )?;
    let conv = rust_mlx::dense::silu(&mlx_rs::ops::conv1d(&inp, &g.conv, 1, 0, 1, 10240)?)?;
    compare("conv", &conv, &o)?;
    use mlx_rs::ops::indexing::IndexOp;
    let q = conv.index((.., .., ..2048)).reshape(&[1, 10, 16, 128])?;
    let k = conv
        .index((.., .., 2048..4096))
        .reshape(&[1, 10, 16, 128])?;
    let v = conv.index((.., .., 4096..)).reshape(&[1, 10, 48, 128])?;
    let eps = rust_mlx::dense::scalar_like(&q, 1e-6)?;
    let q = q
        .multiply(q.square()?.sum_axis(-1, true)?.add(&eps)?.rsqrt()?)?
        .multiply(rust_mlx::dense::scalar_like(&q, 128f32.powf(-0.5))?)?;
    let k = k.multiply(k.square()?.sum_axis(-1, true)?.add(eps)?.rsqrt()?)?;
    compare("q", &q, &o)?;
    compare("k", &k, &o)?;
    compare("v", &v, &o)?;
    let aa = g.a.forward_rows(&mixed)?;
    let bb = g.b.forward_rows(&mixed)?;
    compare("a", &aa, &o)?;
    compare("b", &bb, &o)?;
    let decay = rust_mlx::compiled::decay(&g.alog, &aa, &g.dt)?;
    let beta = mlx_rs::ops::sigmoid(&bb)?;
    compare("g", &decay, &o)?;
    compare("beta", &beta, &o)?;
    let (yy, state) = rust_mlx::gdn_kernel::recurrent(
        &q,
        &k,
        &v,
        &decay,
        &beta,
        &mlx_rs::ops::zeros_dtype(&[1, 48, 128, 128], Dtype::Float32)?,
    )?;
    compare("y", &yy, &o)?;
    compare("state", &state, &o)?;
    let z = g.z.forward(&mixed)?.reshape(&[1, 10, 48, 128])?;
    let norm = mlx_rs::fast::rms_norm(&yy, Some(&g.norm), g.eps)?
        .as_dtype(Dtype::Float32)?
        .multiply(mlx_rs::ops::sigmoid(z.as_dtype(Dtype::Float32)?)?)?
        .as_dtype(yy.dtype())?;
    compare("norm", &norm, &o)?;
    let a = g.forward(&mixed, &mut GdnCache::default())?;
    compare("gdn", &a, &o)?;
    let h = ah.write(x, &a, inject.as_ref().unwrap())?;
    compare("after_attention", &h, &o)?;
    let xn = rust_mlx::hybrid::grouped_norm(&h, &mh.scale, 4, mh.eps)?;
    compare("mlp_normed", &xn, &o)?;
    let lo = rust_mlx::compiled::activate(&mh.down.forward(&xn)?, 4)?;
    compare("mlp_lowrank", &lo, &o)?;
    let up = mh.up.forward(&lo)?;
    compare("mlp_projected", &up, &o)?;
    let (mixin, inj) = mh.forward(&h)?;
    compare("mlp_input", &mixin, &o)?;
    compare("mlp_inject", inj.as_ref().unwrap(), &o)?;
    let y = m.forward(&mixin)?;
    compare("moe", &y, &o)?;
    let z = mh.write(&h, &y, inj.as_ref().unwrap())?;
    compare("output", &z, &o)?;
    Ok(())
}
