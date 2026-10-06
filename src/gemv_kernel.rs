//! Exact singleton GEMV arithmetic with matrix loads shared across short rows.
use crate::{
    metal::{Kernel, Launch, Template},
    weights::Linear,
};
use anyhow::Result;
use mlx_rs::{Array, Dtype};
use std::cell::{Cell, RefCell};
thread_local! {
    static ENABLED: Cell<bool> = Cell::new(std::env::var("RUST_MLX_ROWS_GEMV").map(|v| !matches!(v.to_ascii_lowercase().as_str(),"0"|"false"|"off")).unwrap_or(false));
    static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) };
    static LAUNCHES: Cell<usize> = const { Cell::new(0) };
    static NARROW: Cell<bool> = Cell::new(std::env::var("RUST_MLX_GEMV_NARROW").is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(),"1"|"true"|"on")));
}
pub fn narrow() -> bool {
    NARROW.with(Cell::get)
}
pub fn set_narrow(value: bool) {
    NARROW.with(|x| x.set(value));
}
pub fn launches() -> usize {
    LAUNCHES.with(Cell::get)
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|x| x.set(value));
}
pub(crate) fn project(l: &Linear, x: &Array) -> Result<Option<Array>> {
    if l.quant.is_some()
        || l.weight.ndim() != 2
        || x.ndim() != 3
        || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float32)
        || x.dtype() != l.weight.dtype()
    {
        return Ok(None);
    }
    let (n, k) = (l.weight.shape()[0], l.weight.shape()[1]);
    let Some(rows) = x.shape()[0].checked_mul(x.shape()[1]) else {
        return Ok(None);
    };
    // This is only the native BM4/BN1/SN32 configuration, with no tail blocks.
    if !(2..=8).contains(&rows)
        || !(64..4096).contains(&n)
        || n % 16 != 0
        || k <= 64
        || k % 128 != 0
        || k >= 16 * n
        || x.shape()[2] != k
        || !crate::metal::is_evaluated_row_contiguous(&l.weight)?
    {
        return Ok(None);
    }
    let x = x.contiguous()?;
    LAUNCHES.with(|n| n.set(n.get().wrapping_add(1)));
    KERNEL.with(|cell| -> Result<_> {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::new(
                "rust_mlx_shared_gemv",
                &["x", "w"],
                &["y"],
                include_str!("../kernels/shared_gemv.metal"),
            )?);
        }
        let shape = [x.shape()[0], x.shape()[1], n];
        let mut y = cell
            .as_ref()
            .expect("initialized kernel")
            .launch(Launch {
                inputs: &[&x, &l.weight],
                templates: &[
                    Template::Dtype("T", x.dtype()),
                    Template::Int("ROWS", rows),
                    Template::Int("K_SIZE", k),
                    Template::Int("N_SIZE", n),
                    Template::Int("COLS", if narrow() { 1 } else { 4 }),
                ],
                outputs: &[(&shape, x.dtype())],
                grid: [32 * (n / (4 * if narrow() { 1 } else { 4 })), 1, 4],
                group: [32, 1, 4],
            })?
            .remove(0);
        if let Some(b) = &l.bias {
            y = y.add(b)?;
        }
        Ok(Some(y))
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;
    #[test]
    fn shared_unquantized_rows_preserve_native_singleton_tree() {
        for narrow_mode in [false, true] {
            set_narrow(narrow_mode);
            for dtype in [Dtype::Bfloat16, Dtype::Float32] {
                for (n, k) in [(64, 128), (64, 512), (512, 2560)] {
                    let l = Linear {
                        weight: Array::from_iter(
                            (0..n * k).map(|i| (i as f32 * 0.017).sin() * 0.03),
                            &[n, k],
                        )
                        .as_dtype(dtype)
                        .unwrap(),
                        scales: None,
                        biases: None,
                        bias: None,
                        quant: None,
                    };
                    l.weight.eval().unwrap();
                    for rows in 2..=8 {
                        let x = Array::from_iter(
                            (0..rows * k).map(|i| (i as f32 * 0.031).cos()),
                            &[1, rows, k],
                        )
                        .as_dtype(dtype)
                        .unwrap();
                        let a = project(&l, &x)
                            .unwrap()
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap()
                            .contiguous()
                            .unwrap();
                        set_enabled(false);
                        let b = l
                            .forward_rows(&x)
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap()
                            .contiguous()
                            .unwrap();
                        mlx_rs::transforms::eval([&a, &b]).unwrap();
                        assert_eq!(
                            a.as_slice::<f32>(),
                            b.as_slice::<f32>(),
                            "dtype={dtype:?} n={n} k={k} rows={rows}"
                        );
                        let batch = x.reshape(&[rows, 1, k]).unwrap();
                        let a = project(&l, &batch).unwrap().unwrap();
                        let b = b.reshape(&[rows, 1, n]).unwrap();
                        let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
                        mlx_rs::transforms::eval([&a, &b]).unwrap();
                        assert_eq!(a.as_slice::<f32>(), b.as_slice::<f32>());
                        // Transposed weights must retain their distinct native path.
                        let strided = Linear {
                            weight: l.weight.transpose_axes(&[1, 0]).unwrap(),
                            scales: None,
                            biases: None,
                            bias: None,
                            quant: None,
                        };
                        assert!(
                            project(&strided, &x.index((.., .., ..n.min(k))))
                                .unwrap()
                                .is_none()
                        );
                        strided.weight.eval().unwrap();
                        assert!(
                            project(&strided, &x.index((.., .., ..n.min(k))))
                                .unwrap()
                                .is_none()
                        );
                    }
                }
            }
        }
        set_narrow(false);
    }
}
