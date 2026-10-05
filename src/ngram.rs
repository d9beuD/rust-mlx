use crate::weights::{Quantization, Weights};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use mlx_rs::{Array, ops};
use serde::Deserialize;
use std::{collections::HashMap, fs::File, io::Read, path::Path};

#[derive(Clone, Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}
struct MappedShard {
    map: Mmap,
    base: usize,
    tensors: HashMap<String, TensorMeta>,
}
impl MappedShard {
    fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut n = [0u8; 8];
        file.read_exact(&mut n)?;
        let n = usize::try_from(u64::from_le_bytes(n))?;
        ensure!(n < 100 * 1024 * 1024, "oversized safetensors header");
        let mut header = vec![0; n];
        file.read_exact(&mut header)?;
        let mut json: serde_json::Value = serde_json::from_slice(&header)?;
        json.as_object_mut()
            .context("invalid header")?
            .remove("__metadata__");
        let tensors: HashMap<String, TensorMeta> = serde_json::from_value(json)?;
        // SAFETY: read-only mapping of a completed checkpoint. This engine never
        // writes/truncates the file; Mmap owns its lifetime and every row is bounds-checked.
        let map = unsafe { MmapOptions::new().map(&file)? };
        for meta in tensors.values() {
            ensure!(
                meta.data_offsets[0] <= meta.data_offsets[1]
                    && n + 8 + meta.data_offsets[1] <= map.len(),
                "truncated checkpoint {}",
                path.display()
            );
        }
        Ok(Self {
            map,
            base: n + 8,
            tensors,
        })
    }
    fn row(&self, name: &str, row: usize) -> Result<(&[u8], &TensorMeta)> {
        let m = self.tensors.get(name).context("missing mapped tensor")?;
        ensure!(
            m.shape.len() == 2 && row < m.shape[0],
            "invalid table row {row}"
        );
        let row_bytes = (m.data_offsets[1] - m.data_offsets[0]) / m.shape[0];
        let start = self.base + m.data_offsets[0] + row * row_bytes;
        Ok((&self.map[start..start + row_bytes], m))
    }
}
struct EmbeddingShard {
    file: usize,
    prefix: String,
    rows: usize,
    quant: Option<Quantization>,
}
pub struct NGramTable {
    files: Vec<MappedShard>,
    shards: Vec<EmbeddingShard>,
    offsets: Vec<usize>,
    pub dim: i32,
    batch_mode: std::cell::Cell<bool>,
}
impl NGramTable {
    pub fn load(path: &Path, prefix: &str, config: &serde_json::Value) -> Result<Self> {
        let mut files = Vec::new();
        let mut names: HashMap<String, usize> = HashMap::new();
        let mut paths = std::fs::read_dir(path)?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let file = MappedShard::load(&path)?;
            let index = files.len();
            for n in file.tensors.keys() {
                if n.starts_with(prefix) {
                    names.insert(n.clone(), index);
                }
            }
            files.push(file);
        }
        let mut shards = Vec::new();
        let mut offsets = vec![0usize];
        let mut dim = 0;
        for i in 0.. {
            let p = format!("{prefix}.shards.{i}");
            let key = format!("{p}.weight");
            let Some(&file) = names.get(&key) else { break };
            let m = &files[file].tensors[&key];
            ensure!(m.shape.len() == 2, "invalid ngram embedding shape");
            let q = if names.contains_key(&format!("{p}.scales")) {
                let q = &config["quantization"];
                let q: Quantization = serde_json::from_value(q.get(&p).unwrap_or(q).clone())?;
                ensure!(
                    q.mode == "affine"
                        && [2, 3, 4, 5, 6, 8].contains(&q.bits)
                        && [32, 64, 128].contains(&q.group_size),
                    "unsupported ngram quantization"
                );
                Some(q)
            } else {
                None
            };
            let d = if let Some(q) = &q {
                m.shape[1] * 32 / q.bits as usize
            } else {
                m.shape[1]
            };
            if i == 0 {
                dim = d as i32;
            } else {
                ensure!(dim == d as i32, "ngram row width mismatch");
            }
            offsets.push(offsets.last().copied().unwrap_or(0) + m.shape[0]);
            shards.push(EmbeddingShard {
                file,
                prefix: p,
                rows: m.shape[0],
                quant: q,
            });
        }
        ensure!(!shards.is_empty(), "no ngram shards for {prefix}");
        Ok(Self {
            files,
            shards,
            offsets,
            dim,
            batch_mode: std::cell::Cell::new(std::env::var_os("RUST_MLX_BATCH_PLE").is_some()),
        })
    }
    pub fn set_batch(&self, enabled: bool) {
        self.batch_mode.set(enabled);
    }
    pub fn gather(&self, rows: &[u32], shape: &[i32]) -> Result<Array> {
        if self.batch_mode.get()
            && let Some(y) = self.gather_batch(rows, shape)?
        {
            return Ok(y);
        }
        self.gather_reference(rows, shape)
    }
    /// Combine identically quantized lookup rows into one native dequantization.
    /// Mixed formats fall back to the per-row reference.
    pub fn gather_batch(&self, rows: &[u32], shape: &[i32]) -> Result<Option<Array>> {
        ensure!(!rows.is_empty(), "empty ngram lookup");
        let Some(q) = self.shards[0].quant.as_ref() else {
            return Ok(None);
        };
        if self.shards.iter().any(|s| {
            s.quant
                .as_ref()
                .is_none_or(|v| v.bits != q.bits || v.group_size != q.group_size)
        }) {
            return Ok(None);
        }
        let width = self.dim as usize * q.bits as usize / 32;
        let groups = self.dim as usize / q.group_size as usize;
        let mut packed = Vec::with_capacity(rows.len() * width);
        let mut scales = Vec::with_capacity(rows.len() * groups);
        let mut biases = Vec::with_capacity(rows.len() * groups);
        for &row in rows {
            let row = row as usize;
            ensure!(
                row < *self.offsets.last().context("no table offsets")?,
                "ngram row out of bounds"
            );
            let i = self.offsets.partition_point(|&o| o <= row) - 1;
            let s = &self.shards[i];
            let local = row - self.offsets[i];
            let file = &self.files[s.file];
            let (w, m) = file.row(&format!("{}.weight", s.prefix), local)?;
            let (sc, sm) = file.row(&format!("{}.scales", s.prefix), local)?;
            let (bi, bm) = file.row(&format!("{}.biases", s.prefix), local)?;
            if m.dtype != "U32" || sm.dtype != "BF16" || bm.dtype != "BF16" {
                return Ok(None);
            }
            ensure!(
                w.len() == width * 4 && sc.len() == groups * 2 && bi.len() == groups * 2,
                "invalid packed table width"
            );
            packed.extend(w.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
            scales.extend(
                sc.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| bf16::from_bits(u16::from_le_bytes(*c))),
            );
            biases.extend(
                bi.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| bf16::from_bits(u16::from_le_bytes(*c))),
            );
        }
        let n = rows.len() as i32;
        let y = ops::dequantize(
            Array::from_slice(&packed, &[n, width as i32]),
            Array::from_slice(&scales, &[n, groups as i32]),
            Some(&Array::from_slice(&biases, &[n, groups as i32])),
            q.group_size,
            q.bits,
        )?
        .reshape(shape)?;
        Ok(Some(y))
    }
    pub fn gather_reference(&self, rows: &[u32], shape: &[i32]) -> Result<Array> {
        ensure!(!rows.is_empty(), "empty ngram lookup");
        let mut result = Vec::with_capacity(rows.len());
        for &row in rows {
            let row = row as usize;
            ensure!(
                row < *self.offsets.last().context("no table offsets")?,
                "ngram row out of bounds"
            );
            let index = self.offsets.partition_point(|&off| off <= row) - 1;
            let shard = &self.shards[index];
            let local = row - self.offsets[index];
            ensure!(local < shard.rows, "invalid shard row");
            let file = &self.files[shard.file];
            let p = &shard.prefix;
            let w = read_row(file, &format!("{p}.weight"), local)?;
            let value = if let Some(q) = &shard.quant {
                let scales = read_row(file, &format!("{p}.scales"), local)?;
                let bias = read_row(file, &format!("{p}.biases"), local)?;
                ops::dequantize(&w, &scales, Some(&bias), q.group_size, q.bits)?
            } else {
                w
            };
            result.push(value);
        }
        Ok(ops::concatenate(&result, 0)?.reshape(shape)?)
    }
}
fn read_row(file: &MappedShard, name: &str, row: usize) -> Result<Array> {
    let (bytes, m) = file.row(name, row)?;
    let shape = [1, m.shape[1] as i32];
    Ok(match m.dtype.as_str() {
        "U32" => Array::from_slice(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            &shape,
        ),
        "BF16" => Array::from_slice(
            &bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<_>>(),
            &shape,
        ),
        "F16" => Array::from_slice(
            &bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect::<Vec<_>>(),
            &shape,
        ),
        "F32" => Array::from_slice(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            &shape,
        ),
        dtype => anyhow::bail!("unsupported ngram dtype {dtype}"),
    })
}

pub struct NGramHasher {
    pub multipliers: Vec<i64>,
    pub offsets: Vec<i64>,
    pub sizes: Vec<i64>,
    pub ngram: usize,
    pub heads: usize,
    pub eos: u32,
}
impl NGramHasher {
    pub fn load(w: &Weights, p: &str, ngram: usize, heads: usize, eos: u32) -> Result<Self> {
        let read = |n: &str| -> Result<Vec<i64>> {
            let a = w.tensor(&format!("{p}.{n}"))?;
            a.eval()?;
            Ok(a.as_slice::<i64>().to_vec())
        };
        let s = Self {
            multipliers: read("layer_multipliers")?,
            offsets: read("ngram_heads_offsets")?,
            sizes: read("ngram_heads_vocab_sizes")?,
            ngram,
            heads,
            eos,
        };
        ensure!(
            s.multipliers.len() == ngram
                && s.offsets.len() == (ngram - 1) * heads
                && s.sizes.len() == s.offsets.len(),
            "ngram constants mismatch"
        );
        ensure!(
            s.sizes.iter().all(|&s| s > 0),
            "ngram vocab sizes must be positive"
        );
        Ok(s)
    }
    pub fn rows(&self, tokens: &[u32], history: &[u32]) -> Result<Vec<u32>> {
        let mut stream = history.to_vec();
        stream.extend_from_slice(tokens);
        let mut rows = Vec::with_capacity(tokens.len() * self.sizes.len());
        for t in history.len()..stream.len() {
            let mut context = Vec::with_capacity(self.ngram);
            context.push(stream[t]);
            let mut cut = false;
            for ago in 1..self.ngram {
                let raw = t.checked_sub(ago).map(|i| stream[i]);
                cut |= raw.is_none_or(|id| id == self.eos);
                context.push(if cut {
                    self.eos
                } else {
                    raw.unwrap_or(self.eos)
                });
            }
            let mut mixed = (context[0] as u64).wrapping_mul(self.multipliers[0] as u64);
            for n in 2..=self.ngram {
                mixed ^= (context[n - 1] as u64).wrapping_mul(self.multipliers[n - 1] as u64);
                for head in 0..self.heads {
                    let h = (n - 2) * self.heads + head;
                    let row = (mixed % self.sizes[h] as u64) + self.offsets[h] as u64;
                    rows.push(u32::try_from(row)?);
                }
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_chunking_eos_and_overflow() {
        let h = NGramHasher {
            multipliers: vec![i64::MAX, 3571, 9209],
            offsets: vec![0, 101, 202, 305],
            sizes: vec![101, 101, 103, 107],
            ngram: 3,
            heads: 2,
            eos: 7,
        };
        let tokens = [1, 2, 7, 3, 4, 5];
        let full = h.rows(&tokens, &[]).unwrap();
        let mut chunk = h.rows(&tokens[..3], &[]).unwrap();
        chunk.extend(h.rows(&tokens[3..], &tokens[1..3]).unwrap());
        assert_eq!(full, chunk);
        assert_eq!(&full[3 * 4..4 * 4], h.rows(&[3], &[7]).unwrap());
    }
}
