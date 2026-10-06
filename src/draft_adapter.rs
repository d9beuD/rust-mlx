//! Activation-dependent private MTP correction; original target/head are immutable.
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype, ops};
use std::path::Path;

pub struct DraftAdapter {
    pub a: Array,
    pub b: Array,
    pub hidden: i32,
    pub hc: i32,
}
impl DraftAdapter {
    pub fn load(model: &Path, artifact: &Path, hidden: i32, hc: i32) -> Result<Self> {
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifact.join("metadata.json"))?)?;
        let weights = artifact.join("adapter.safetensors");
        ensure!(
            m["complete"] == true && m["kind"] == "mtp-residual-v1",
            "incomplete/unsupported adapter"
        );
        for (key, path) in [
            ("config_sha256", model.join("config.json")),
            ("tokenizer_sha256", model.join("tokenizer.json")),
            ("adapter_sha256", weights.clone()),
        ] {
            ensure!(
                m[key].as_str() == Some(crate::resident_quant::sha256_file(&path)?.as_str()),
                "adapter identity mismatch: {key}"
            );
        }
        let shards = m["checkpoint_sha256"]
            .as_object()
            .context("missing checkpoint hashes")?;
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model.join("model.safetensors.index.json"))?)?;
        let expected = index["weight_map"]
            .as_object()
            .context("missing checkpoint index")?
            .values()
            .map(|v| v.as_str().context("invalid shard"))
            .collect::<Result<std::collections::HashSet<_>>>()?;
        ensure!(
            shards.len() == expected.len(),
            "incomplete checkpoint identity"
        );
        for name in expected {
            ensure!(
                std::path::Path::new(name)
                    .file_name()
                    .and_then(|n| n.to_str())
                    == Some(name)
                    && shards[name].as_str()
                        == Some(crate::resident_quant::sha256_file(&model.join(name))?.as_str()),
                "checkpoint identity mismatch"
            );
        }
        let mut tensors = Array::load_safetensors(weights)?;
        let a = tensors.remove("a").context("missing adapter a")?;
        let b = tensors.remove("b").context("missing adapter b")?;
        ensure!(tensors.is_empty(), "unexpected adapter tensors");
        Self::new(a, b, hidden, hc)
    }
    pub fn new(a: Array, b: Array, hidden: i32, hc: i32) -> Result<Self> {
        ensure!(
            hidden > 0
                && hc > 0
                && a.ndim() == 2
                && b.ndim() == 2
                && a.shape()[1] == hidden * (hc + 2)
                && b.shape() == [hidden, a.shape()[0]]
                && (1..=64).contains(&a.shape()[0])
                && a.dtype() == Dtype::Float32
                && b.dtype() == Dtype::Float32,
            "adapter shape/dtype mismatch"
        );
        let a = a.contiguous()?;
        let b = b.contiguous()?;
        for tensor in [&a, &b] {
            tensor.eval()?;
            ensure!(
                tensor.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "nonfinite adapter"
            );
        }
        Ok(Self { a, b, hidden, hc })
    }
    pub fn features(mixed: &Array, previous: &Array, embedding: &Array) -> Result<Array> {
        let normalize = |x: &Array| -> Result<Array> {
            let x = x.as_dtype(Dtype::Float32)?;
            let rms = x
                .square()?
                .mean_axis(-1, true)?
                .add(Array::from_f32(1e-6))?
                .sqrt()?;
            Ok(x.divide(rms)?)
        };
        Ok(ops::concatenate(
            &[
                normalize(mixed)?,
                normalize(previous)?,
                normalize(embedding)?,
            ],
            -1,
        )?)
    }
    pub fn apply(&self, mixed: &Array, previous: &Array, embedding: &Array) -> Result<Array> {
        ensure!(
            mixed.ndim() == 3
                && mixed.shape()[2] == self.hidden
                && previous.shape() == [mixed.shape()[0], mixed.shape()[1], self.hidden * self.hc]
                && embedding.shape() == mixed.shape(),
            "adapter input mismatch"
        );
        let x = Self::features(mixed, previous, embedding)?;
        let correction = x
            .matmul(self.a.transpose()?)?
            .matmul(self.b.transpose()?)?
            .as_dtype(mixed.dtype())?;
        Ok(mixed.add(correction)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_adapter_identity_and_invalid_weights() {
        let mixed = Array::from_slice(&[1.0f32, -2.0], &[1, 1, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let previous = Array::from_slice(&[3.0f32, 4.0, 1.0, -1.0], &[1, 1, 4]);
        let a = ops::zeros::<f32>(&[2, 8]).unwrap();
        let b = ops::zeros::<f32>(&[2, 2]).unwrap();
        let adapter = DraftAdapter::new(a, b, 2, 2).unwrap();
        let actual = adapter
            .apply(&mixed, &previous, &mixed)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        actual.eval().unwrap();
        assert_eq!(actual.as_slice::<f32>(), &[1.0, -2.0]);
        assert!(
            DraftAdapter::new(
                Array::from_slice(&[f32::NAN; 8], &[1, 8]),
                ops::zeros::<f32>(&[2, 1]).unwrap(),
                2,
                2
            )
            .is_err()
        );
        assert!(
            DraftAdapter::new(
                ops::zeros::<f32>(&[2, 7]).unwrap(),
                ops::zeros::<f32>(&[2, 2]).unwrap(),
                2,
                2
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;
    #[test]
    fn artifact_identity_rejects_changed_checkpoint_and_missing_parameters() {
        let root = std::env::temp_dir().join(format!(
            "rust-mlx-adapter-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model = root.join("model");
        let artifact = root.join("artifact");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&artifact).unwrap();
        for (name, bytes) in [
            ("config.json", "{}"),
            ("tokenizer.json", "{}"),
            ("shard.safetensors", "checkpoint"),
            (
                "model.safetensors.index.json",
                "{\"weight_map\":{\"head.weight\":\"shard.safetensors\"}}",
            ),
        ] {
            std::fs::write(model.join(name), bytes).unwrap();
        }
        let a = ops::zeros::<f32>(&[2, 8]).unwrap();
        let b = ops::zeros::<f32>(&[2, 2]).unwrap();
        let path = artifact.join("adapter.safetensors");
        Array::save_safetensors([("a", &a), ("b", &b)], None, &path).unwrap();
        let sha = |p: &Path| crate::resident_quant::sha256_file(p).unwrap();
        let mut metadata = serde_json::json!({"complete":true,"kind":"mtp-residual-v1","config_sha256":sha(&model.join("config.json")),"tokenizer_sha256":sha(&model.join("tokenizer.json")),"adapter_sha256":sha(&path),"checkpoint_sha256":{"shard.safetensors":sha(&model.join("shard.safetensors"))}});
        let write = |m: &serde_json::Value| {
            std::fs::write(
                artifact.join("metadata.json"),
                serde_json::to_vec(m).unwrap(),
            )
            .unwrap()
        };
        write(&metadata);
        assert!(DraftAdapter::load(&model, &artifact, 2, 2).is_ok());
        std::fs::write(model.join("shard.safetensors"), "changed").unwrap();
        assert!(DraftAdapter::load(&model, &artifact, 2, 2).is_err());
        metadata["checkpoint_sha256"]["shard.safetensors"] =
            sha(&model.join("shard.safetensors")).into();
        Array::save_safetensors([("a", &a)], None, &path).unwrap();
        metadata["adapter_sha256"] = sha(&path).into();
        write(&metadata);
        assert!(DraftAdapter::load(&model, &artifact, 2, 2).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
