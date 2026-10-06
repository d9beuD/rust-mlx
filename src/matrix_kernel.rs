//! Matrix-unit feasibility experiments, disabled unless explicitly requested.
//! TensorOps changes weight rounding/reduction; callers must measure exactness.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Dtype, ops};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    process::Command,
    sync::OnceLock,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InputMode {
    Staged,
    Registers,
    AffineRegisters,
    CompactRegisters,
    AffineCompact,
    Packed,
}
thread_local! {
    static KERNELS: RefCell<HashMap<(InputMode, i32), Kernel>> = RefCell::new(HashMap::new());
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static PACKED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
/// A conservative host guard for these M5/macOS27 research kernels.
pub fn supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        let read = |name: &str, args: &[&str]| {
            Command::new(name)
                .args(args)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        };
        read("sysctl", &["-n", "machdep.cpu.brand_string"])
            .is_some_and(|s| s.starts_with("Apple M5"))
            && read("sw_vers", &["-productVersion"])
                .and_then(|s| s.split('.').next()?.parse::<u32>().ok())
                .is_some_and(|n| n >= 27)
    })
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
/// Only research CLIs enable this; component parity is insufficient for defaults.
pub fn set_enabled(value: bool) {
    ENABLED.with(|v| v.set(value));
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}
pub fn packed() -> bool {
    PACKED.with(Cell::get)
}
pub fn set_packed(value: bool) {
    PACKED.with(|v| v.set(value));
}
pub(crate) fn selected(l: &Linear, x: &Array) -> Result<Option<Array>> {
    // Cooperative-input prototypes have a measured numerical counterexample
    // under the installed shader validator. Keep that selected path native;
    // explicit component project() remains available to reproduce the failure.
    if !packed() && instrumented() {
        return Ok(None);
    }
    // Q8 already fails the strict actual head test. Limit the full-model
    // experiment to supported multi-row Q4/Q5/Q6 resident projections.
    if !enabled()
        || !crate::verification::rows()
        || x.ndim() != 3
        || !(2..=4).contains(&x.shape()[1])
        || l.weight.ndim() != 2
        || l.weight.shape()[0] < 2048
        || !l
            .quant
            .as_ref()
            .is_some_and(|q| [4, 5, 6].contains(&q.bits))
    {
        return Ok(None);
    }
    if packed() && !l.quant.as_ref().is_some_and(|q| q.bits == 4) {
        return Ok(None);
    }
    let out = project(
        l,
        x,
        if packed() {
            InputMode::Packed
        } else {
            InputMode::AffineCompact
        },
        8,
    )?;
    if out.is_some() {
        CALLS.with(|n| n.set(n.get().wrapping_add(1)));
    }
    Ok(out)
}
pub fn instrumented() -> bool {
    // The validator is a process-start setting, not a mutable dispatch policy.
    static INSTRUMENTED: OnceLock<bool> = OnceLock::new();
    *INSTRUMENTED.get_or_init(|| std::env::var("MTL_SHADER_VALIDATION").is_ok_and(|v| v == "1"))
}

/// Explicit experiment with 16 padded input rows and bounded packed loads.
pub fn project(l: &Linear, x: &Array, mode: InputMode, splits: i32) -> Result<Option<Array>> {
    if !supported() {
        return Ok(None);
    }
    let Some(q) = &l.quant else { return Ok(None) };
    if q.mode != "affine"
        || ![4, 5, 6, 8].contains(&q.bits)
        || ![32, 64, 128].contains(&q.group_size)
        || x.ndim() != 3
        || x.shape()[0] != 1
        || !(1..=4).contains(&x.shape()[1])
        || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        || l.weight.ndim() != 2
        || l.weight.dtype() != Dtype::Uint32
        || !(if matches!(mode, InputMode::AffineRegisters | InputMode::AffineCompact) {
            splits
                == if mode == InputMode::AffineCompact {
                    8
                } else {
                    32
                }
                && x.shape()[2] % 512 == 0
        } else if mode == InputMode::Packed {
            [2, 4, 8, 16].contains(&splits) && q.bits == 4 && x.shape()[2] % 512 == 0
        } else {
            [1, 2, 4, 8].contains(&splits)
        })
    {
        return Ok(None);
    }
    let (t, k, n) = (x.shape()[1], x.shape()[2], l.weight.shape()[0]);
    if k <= 0
        || n <= 0
        || k % (if mode == InputMode::CompactRegisters {
            64
        } else {
            16
        } * splits)
            != 0
        || k % q.group_size != 0
        || n % 32 != 0
        || k as i64 * n as i64 > i32::MAX as i64
    {
        return Ok(None);
    }
    let (Some(sc), Some(bs)) = (&l.scales, &l.biases) else {
        return Ok(None);
    };
    if sc.dtype() != x.dtype() || bs.dtype() != x.dtype() {
        return Ok(None);
    }
    ensure!(
        l.weight.shape()[1] as i64 * 32 == k as i64 * q.bits as i64
            && sc.shape() == [n, k / q.group_size]
            && bs.shape() == sc.shape(),
        "invalid matrix-unit projection geometry"
    );
    let padded = if matches!(
        mode,
        InputMode::AffineRegisters | InputMode::AffineCompact | InputMode::Packed
    ) {
        x.contiguous()?
    } else {
        let padding = ops::zeros_dtype(
            &[
                1,
                if mode == InputMode::CompactRegisters {
                    8
                } else {
                    16
                } - t,
                k,
            ],
            x.dtype(),
        )?;
        ops::concatenate(&[x, &padding], 1)?.contiguous()?
    };
    let key = (mode, splits);
    KERNELS.with(|cell| -> Result<_> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(key) {
            entry.insert(Kernel::with_header(
                &format!("rust_mlx_matrix_{mode:?}_split{splits}"),
                &["x", "w", "scales", "biases"],
                &["y"],
                if mode == InputMode::Packed {
                    include_str!("../kernels/matrix_packed.metal")
                } else if matches!(mode, InputMode::AffineRegisters | InputMode::AffineCompact) {
                    include_str!("../kernels/matrix_affine.metal")
                } else {
                    include_str!("../kernels/matrix_verify.metal")
                },
                include_str!("../kernels/matrix_verify.h"),
            )?);
        }
        let shape = [1, t, n];
        let mut y = kernels
            .get(&key)
            .context("matrix kernel missing")?
            .launch(Launch {
                inputs: &[&padded, &l.weight, sc, bs],
                templates: &[
                    Template::Dtype("T", x.dtype()),
                    Template::Int("VERIFY_T", t),
                    Template::Int("K_SIZE", k),
                    Template::Int("N_SIZE", n),
                    Template::Int("BITS", q.bits),
                    Template::Int("GROUP_SIZE", q.group_size),
                    Template::Int("SPLITS", splits),
                    Template::Int(
                        "TILE_M",
                        if mode == InputMode::CompactRegisters {
                            8
                        } else {
                            16
                        },
                    ),
                    Template::Int(
                        "TILE_K",
                        if mode == InputMode::CompactRegisters {
                            64
                        } else {
                            16
                        },
                    ),
                    Template::Bool(
                        "REGISTER_INPUT",
                        matches!(mode, InputMode::Registers | InputMode::CompactRegisters),
                    ),
                    Template::Bool("HYBRID_INPUT", mode == InputMode::CompactRegisters),
                ],
                outputs: &[(&shape, x.dtype())],
                grid: [splits * 32, n / 32, 1],
                group: [splits * 32, 1, 1],
            })?
            .remove(0);
        if let Some(bias) = &l.bias {
            y = y.add(bias)?;
        }
        Ok(Some(y))
    })
}
