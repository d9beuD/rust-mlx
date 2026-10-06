//! Verify every projected block maximum and index, then the final greedy ID.
use anyhow::{Result, ensure};
use mlx_rs::{
    Array, Dtype, ops,
    ops::indexing::{self, IndexOp},
};
use rust_mlx::{
    greedy_head, qmv_kernel,
    weights::{Linear, Quantization, Weights},
};

fn compare(head: &Linear, x: &Array) -> Result<()> {
    qmv_kernel::set_enabled(false);
    let logits = head.forward_rows(x)?;
    let (b, t, n) = (x.shape()[0], x.shape()[1], head.weight.shape()[0]);
    let blocks = logits.reshape(&[b * t, n / 8, 8])?;
    let values = blocks
        .max_axis(-1, false)?
        .as_dtype(Dtype::Float32)?
        .contiguous()?;
    let local = indexing::argmax_axis(&blocks, -1, false)?;
    let ids = local
        .add(Array::from_iter(
            (0..n / 8).map(|i| i as u32 * 8),
            &[1, n / 8],
        ))?
        .contiguous()?;
    let (v, i) = greedy_head::diagnostic_partials(head, x)?
        .ok_or_else(|| anyhow::anyhow!("candidate not engaged"))?;
    let (v, i) = (v.contiguous()?, i.contiguous()?);
    mlx_rs::transforms::eval([&values, &ids, &v, &i])?;
    ensure!(
        values.as_slice::<f32>() == v.as_slice::<f32>(),
        "block values differ"
    );
    ensure!(
        ids.as_slice::<u32>() == i.as_slice::<u32>(),
        "block indices differ"
    );
    greedy_head::set_enabled(true);
    let actual = greedy_head::greedy(head, x)?.contiguous()?;
    let expected = indexing::argmax_axis(&logits, -1, false)?.contiguous()?;
    mlx_rs::transforms::eval([&actual, &expected])?;
    ensure!(
        actual.as_slice::<u32>() == expected.as_slice::<u32>(),
        "greedy IDs differ"
    );
    greedy_head::set_enabled(false);
    qmv_kernel::set_enabled(true);
    Ok(())
}

#[test]
fn projection_rounding_ties_and_ids_match_native_bf16() -> Result<()> {
    for group in [32, 64, 128] {
        let (n, k) = (2048, 512);
        let w = Array::from_iter((0..n * k).map(|j| (j as f32 * 0.009).sin() * 0.04), &[n, k])
            .as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = ops::quantize(w, group, 8)?;
        let head = Linear {
            weight,
            scales: Some(scales),
            biases: Some(biases),
            bias: None,
            quant: Some(Quantization {
                bits: 8,
                group_size: group,
                mode: "affine".into(),
            }),
        };
        for (b, t) in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)] {
            let x = Array::from_iter((0..b * t * k).map(|j| (j as f32 * 0.017).cos()), &[b, t, k])
                .as_dtype(Dtype::Bfloat16)?;
            compare(&head, &x)?;
            // Every projected value is zero, so native lowest-global-ID ties must be0.
            compare(&head, &ops::zeros_dtype(&[b, t, k], Dtype::Bfloat16)?)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires target checkpoint"]
fn actual_bf16_vocabulary_head_matches_native_blocks_and_ids() -> Result<()> {
    let p = std::env::var("RUST_MLX_TARGET_MODEL")?;
    let w = Weights::load(std::path::Path::new(&p))?;
    let head = w.linear("language_model.lm_head")?;
    ensure!(
        head.weight.shape() == [248320, 640],
        "actual mixed-quantized head required"
    );
    for t in 1..=4 {
        let x = Array::from_iter(
            (0..t * 2560).map(|j| (j as f32 * 0.017).cos()),
            &[1, t, 2560],
        )
        .as_dtype(Dtype::Bfloat16)?;
        compare(&head, &x)?;
        println!("GREEDY_HEAD_ACTUAL_BLOCKS_IDS_EXACT T={t}");
    }
    // Explicit prefix views retain evaluated source ownership and row-contiguous data.
    compare(
        &head,
        &ops::zeros_dtype(&[1, 1, 2560], Dtype::Bfloat16)?.index((.., .., ..)),
    )?;
    Ok(())
}

#[test]
fn reduction_extremes_and_lowest_global_ties_match_native() -> Result<()> {
    use rust_mlx::metal::{Kernel, Launch, Template};
    let n = 31040;
    let mut data = vec![f32::NEG_INFINITY; n * 4];
    data[n..2 * n].fill(f32::NAN);
    data[n + 777] = 2.0;
    data[2 * n..3 * n].fill(0.0);
    data[2 * n + 50] = f32::INFINITY;
    data[2 * n + 1000] = f32::INFINITY;
    data[3 * n..].fill(-0.0);
    let values = Array::from_slice(&data, &[4, n as i32]);
    let indices = Array::from_iter(
        (0..4).flat_map(|_| (0..n).map(|i| i as u32 * 8)),
        &[4, n as i32],
    );
    let kernel = Kernel::new(
        "rust_mlx_greedy_reduce_extremes",
        &["maxima", "indices"],
        &["tokens"],
        include_str!("../kernels/greedy_reduce.metal"),
    )?;
    let actual = kernel
        .launch(Launch {
            inputs: &[&values, &indices],
            templates: &[Template::Int("PARTIALS", n as i32)],
            outputs: &[(&[4], Dtype::Uint32)],
            grid: [256, 4, 1],
            group: [256, 1, 1],
        })?
        .remove(0);
    let expected =
        indexing::argmax_axis(&values, -1, false)?.multiply(Array::from_slice(&[8u32], &[]))?;
    mlx_rs::transforms::eval([&actual, &expected])?;
    ensure!(
        actual.as_slice::<u32>() == expected.as_slice::<u32>(),
        "extreme argmax differs"
    );
    Ok(())
}
