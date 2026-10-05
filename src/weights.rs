use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, ops};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Quantization {
    pub bits: i32,
    pub group_size: i32,
    #[serde(default = "affine")]
    pub mode: String,
}
fn affine() -> String {
    "affine".into()
}

pub struct Weights {
    pub tensors: HashMap<String, Array>,
    pub config: serde_json::Value,
}
impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        let index_path = path.join("model.safetensors.index.json");
        let mut files = if index_path.exists() {
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path)?)?;
            index["weight_map"]
                .as_object()
                .context("missing weight map")?
                .values()
                .map(|v| v.as_str().context("invalid shard name").map(str::to_owned))
                .collect::<Result<Vec<_>>>()?
        } else {
            std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".safetensors"))
                .collect()
        };
        files.sort();
        files.dedup();
        ensure!(!files.is_empty(), "no safetensors in {}", path.display());
        let mut tensors = HashMap::new();
        for file in files {
            eprintln!("loading {file}");
            let shard = Array::load_safetensors(path.join(&file))
                .with_context(|| format!("loading {file}"))?;
            for (name, tensor) in shard {
                if name.starts_with("visual.")
                    || name.starts_with("vision_tower.")
                    || name.contains(".ngram_embedding.shards.")
                {
                    continue;
                }
                ensure!(
                    tensors.insert(name.clone(), tensor).is_none(),
                    "duplicate tensor {name}"
                );
            }
        }
        Ok(Self { tensors, config })
    }
    pub fn tensor(&self, name: &str) -> Result<Array> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing tensor {name}"))
    }
    pub fn quantization(&self, prefix: &str) -> Result<Option<Quantization>> {
        if !self.tensors.contains_key(&format!("{prefix}.scales")) {
            return Ok(None);
        }
        let q = self
            .config
            .get("quantization")
            .or_else(|| self.config.get("quantization_config"))
            .context("quantized weight without quantization config")?;
        let q: Quantization = serde_json::from_value(q.get(prefix).unwrap_or(q).clone())?;
        ensure!(
            q.mode == "affine",
            "unsupported quantization mode {}",
            q.mode
        );
        ensure!(
            [2, 3, 4, 5, 6, 8].contains(&q.bits) && [32, 64, 128].contains(&q.group_size),
            "unsupported quantization {q:?}"
        );
        Ok(Some(q))
    }
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        Ok(Linear {
            weight: self.tensor(&format!("{prefix}.weight"))?,
            scales: self.tensors.get(&format!("{prefix}.scales")).cloned(),
            biases: self.tensors.get(&format!("{prefix}.biases")).cloned(),
            bias: self.tensors.get(&format!("{prefix}.bias")).cloned(),
            quant: self.quantization(prefix)?,
        })
    }
}

pub struct Linear {
    pub weight: Array,
    pub scales: Option<Array>,
    pub biases: Option<Array>,
    pub bias: Option<Array>,
    pub quant: Option<Quantization>,
}
impl Linear {
    /// Match the oracle's decode-equivalent reductions for narrow gate projections.
    pub fn forward_rows(&self, x: &Array) -> Result<Array> {
        if x.ndim() == 3
            && x.shape()[0] * x.shape()[1] > 1
            && self.quant.is_some()
            && x.dtype() != mlx_rs::Dtype::Float32
        {
            let q = self.quant.as_ref().context("missing quantization")?;
            let ids = ops::zeros_dtype(&[x.shape()[0], x.shape()[1], 1], mlx_rs::Dtype::Int32)?;
            let xe = x.contiguous()?.expand_dims_axes(&[-2, -3])?;
            let mut y = ops::gather_qmm(
                &xe,
                &self.weight.expand_dims(0)?,
                &self
                    .scales
                    .as_ref()
                    .context("missing scales")?
                    .expand_dims(0)?,
                self.biases
                    .as_ref()
                    .map(|v| v.expand_dims(0))
                    .transpose()?
                    .as_ref(),
                None,
                &ids,
                true,
                q.group_size,
                q.bits,
                false,
            )?
            .squeeze_axes(&[-2, -3])?;
            if let Some(b) = &self.bias {
                y = y.add(b)?;
            }
            Ok(y)
        } else if x.ndim() == 3 && x.shape()[0] * x.shape()[1] > 1 {
            use mlx_rs::ops::indexing::IndexOp;
            let flat = x.reshape(&[1, x.shape()[0] * x.shape()[1], x.shape()[2]])?;
            let rows = (0..flat.shape()[1])
                .map(|i| self.forward(&flat.index((.., i..i + 1, ..))))
                .collect::<Result<Vec<_>>>()?;
            let y = ops::concatenate(&rows.iter().collect::<Vec<_>>(), 1)?;
            Ok(y.reshape(&[x.shape()[0], x.shape()[1], y.shape()[2]])?)
        } else {
            self.forward(x)
        }
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if crate::verification::rows() && x.ndim() == 3 && x.shape()[0] * x.shape()[1] > 1 {
            return self.forward_rows(x);
        }

        let mut y = if let Some(q) = &self.quant {
            ops::quantized_matmul(
                x,
                &self.weight,
                self.scales.as_ref().context("missing scales")?,
                self.biases.as_ref(),
                true,
                q.group_size,
                q.bits,
            )?
        } else {
            x.matmul(self.weight.t())?
        };
        if let Some(b) = &self.bias {
            y = y.add(b)?;
        }
        Ok(y)
    }
    pub fn embedding(&self, ids: &Array) -> Result<Array> {
        let w = self.weight.take_axis(ids, 0)?;
        if let Some(q) = &self.quant {
            let scales = self
                .scales
                .as_ref()
                .context("missing scales")?
                .take_axis(ids, 0)?;
            let biases = self
                .biases
                .as_ref()
                .map(|b| b.take_axis(ids, 0))
                .transpose()?;
            Ok(ops::dequantize(
                &w,
                &scales,
                biases.as_ref(),
                q.group_size,
                q.bits,
            )?)
        } else {
            Ok(w)
        }
    }
}
