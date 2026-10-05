//! MLX's native recurrent reduction order, hosted through its retained Metal graph.
use crate::metal::{Kernel, Launch, Template};
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype};
use std::cell::RefCell;
thread_local! {static KERNEL:RefCell<Option<Kernel>>=const {RefCell::new(None)};static HISTORY:RefCell<Option<Kernel>>=const{RefCell::new(None)};}
pub fn recurrent(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    ensure!(q.ndim() == 4 && q.shape() == k.shape(), "invalid GDN q/k");
    let (b, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    ensure!(
        v.ndim() == 4 && v.shape()[0] == b && v.shape()[1] == t,
        "invalid GDN values"
    );
    let (hv, dv) = (v.shape()[2], v.shape()[3]);
    ensure!(
        b > 0
            && t > 0
            && hk > 0
            && dk > 0
            && hv > 0
            && dv > 0
            && dk % 32 == 0
            && hv % hk == 0
            && dv % 4 == 0,
        "unsupported GDN kernel dimensions"
    );
    ensure!(
        g.shape() == [b, t, hv] && beta.shape() == [b, t, hv] && state.shape() == [b, hv, dv, dk],
        "invalid GDN state/gates"
    );
    ensure!(
        q.dtype() == k.dtype() && q.dtype() == v.dtype() && state.dtype() == Dtype::Float32,
        "invalid GDN dtype"
    );
    let time = Array::from_int(t);
    KERNEL.with(|cell| {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::new(
                "rust_mlx_native_gated_delta",
                &["q", "k", "v", "g", "beta", "state_in", "T"],
                &["y", "state_out"],
                include_str!("../kernels/gated_delta.metal"),
            )?);
        }
        let mut out = cell.as_ref().expect("initialized above").launch(Launch {
            inputs: &[q, k, v, g, beta, state, &time],
            templates: &[
                Template::Dtype("InT", q.dtype()),
                Template::Dtype("StT", state.dtype()),
                Template::Int("Dk", dk),
                Template::Int("Dv", dv),
                Template::Int("Hk", hk),
                Template::Int("Hv", hv),
            ],
            outputs: &[(&[b, t, hv, dv], q.dtype()), (state.shape(), state.dtype())],
            grid: [32, dv, b * hv],
            group: [32, 4, 1],
        })?;
        let s = out.pop().expect("two declared outputs");
        let y = out.pop().expect("two declared outputs");
        Ok((y, s))
    })
}

pub fn recurrent_with_history(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array, Array)> {
    ensure!(q.ndim() == 4 && q.shape() == k.shape(), "invalid GDN q/k");
    let (b, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    ensure!(
        v.ndim() == 4 && v.shape()[0] == b && v.shape()[1] == t,
        "invalid GDN values"
    );
    let (hv, dv) = (v.shape()[2], v.shape()[3]);
    ensure!(
        b > 0
            && t > 0
            && hk > 0
            && dk > 0
            && hv > 0
            && dv > 0
            && dk % 32 == 0
            && hv % hk == 0
            && dv % 4 == 0,
        "unsupported GDN kernel dimensions"
    );
    ensure!(
        g.shape() == [b, t, hv] && beta.shape() == [b, t, hv] && state.shape() == [b, hv, dv, dk],
        "invalid GDN state/gates"
    );
    ensure!(
        q.dtype() == k.dtype() && q.dtype() == v.dtype() && state.dtype() == Dtype::Float32,
        "invalid GDN dtype"
    );
    let time = Array::from_int(t);
    HISTORY.with(|cell| {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::new(
                "rust_mlx_native_gated_delta_history",
                &["q", "k", "v", "g", "beta", "state_in", "T"],
                &["y", "state_out", "states"],
                include_str!("../kernels/gated_delta_history.metal"),
            )?);
        }
        let mut out = cell.as_ref().expect("initialized above").launch(Launch {
            inputs: &[q, k, v, g, beta, state, &time],
            templates: &[
                Template::Dtype("InT", q.dtype()),
                Template::Dtype("StT", state.dtype()),
                Template::Int("Dk", dk),
                Template::Int("Dv", dv),
                Template::Int("Hk", hk),
                Template::Int("Hv", hv),
            ],
            outputs: &[
                (&[b, t, hv, dv], q.dtype()),
                (state.shape(), state.dtype()),
                (&[b, t, hv, dv, dk], state.dtype()),
            ],
            grid: [32, dv, b * hv],
            group: [32, 4, 1],
        })?;
        let history = out.pop().expect("three declared outputs");
        let s = out.pop().expect("two declared outputs");
        let y = out.pop().expect("two declared outputs");
        Ok((y, s, history))
    })
}

thread_local! {static PACKED:RefCell<Option<Kernel>>=const{RefCell::new(None)};static PACKED_HISTORY:RefCell<Option<Kernel>>=const{RefCell::new(None)};}
pub fn packed(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    history: bool,
) -> Result<Vec<Array>> {
    ensure!(
        q.ndim() == 4 && q.shape() == k.shape() && v.ndim() == 4,
        "invalid packed GDN tensors"
    );
    let (b, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (hv, dv) = (v.shape()[2], v.shape()[3]);
    ensure!(
        b > 0
            && t > 0
            && hk > 0
            && dk == 128
            && hv > 0
            && hv % hk == 0
            && dv > 0
            && dv % 8 == 0
            && v.shape()[..2] == [b, t],
        "unsupported packed dimensions"
    );
    ensure!(
        g.shape() == [b, t, hv]
            && beta.shape() == [b, t, hv]
            && state.shape() == [b, hv, dv, dk]
            && g.dtype() == Dtype::Float32
            && state.dtype() == Dtype::Float32
            && q.dtype() == k.dtype()
            && q.dtype() == v.dtype(),
        "invalid packed state/gates"
    );
    let run = |cell: &RefCell<Option<Kernel>>| -> Result<Vec<Array>> {
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(Kernel::new(
                if history {
                    "rust_mlx_packed_gdn_history"
                } else {
                    "rust_mlx_packed_gdn"
                },
                &["q", "k", "v", "g", "beta", "state_in", "T"],
                if history {
                    &["y", "state_out", "states"]
                } else {
                    &["y", "state_out"]
                },
                if history {
                    include_str!("../kernels/gated_delta_packed_history.metal")
                } else {
                    include_str!("../kernels/gated_delta_packed.metal")
                },
            )?);
        }
        let ys = [b, t, hv, dv];
        let hs = [b, t, hv, dv, dk];
        let mut outputs = vec![(ys.as_slice(), q.dtype()), (state.shape(), state.dtype())];
        if history {
            outputs.push((hs.as_slice(), state.dtype()));
        }
        cell.as_ref().expect("initialized above").launch(Launch {
            inputs: &[q, k, v, g, beta, state, &Array::from_int(t)],
            templates: &[
                Template::Dtype("InT", q.dtype()),
                Template::Dtype("StT", state.dtype()),
                Template::Int("Dk", dk),
                Template::Int("Dv", dv),
                Template::Int("Hk", hk),
                Template::Int("Hv", hv),
            ],
            outputs: &outputs,
            grid: [32, dv / 8, b * hv],
            group: [32, 2, 1],
        })
    };
    if history {
        PACKED_HISTORY.with(run)
    } else {
        PACKED.with(run)
    }
}
