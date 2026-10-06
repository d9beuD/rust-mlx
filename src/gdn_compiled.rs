//! Experimental pure recurrent attention graph: all tensors are explicit inputs.
use crate::{
    hybrid::{Gdn, GdnCache},
    hyper_compiled::{LinearMeta, append},
};
use anyhow::{Context, Result, ensure};
use mlx_rs::{
    Array, Dtype,
    error::{Exception, Result as MlxResult},
    transforms::compile::compile,
};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};
type Graph = Box<dyn FnMut(&[Array]) -> MlxResult<Vec<Array>>>;
thread_local! {
    static ENABLED: Cell<bool> = Cell::new(std::env::var("RUST_MLX_COMPILE_GDN").is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on")));
    static GRAPHS: RefCell<HashMap<Plan, Graph>> = RefCell::new(HashMap::new());
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn set_enabled(value: bool) {
    ENABLED.with(|x| x.set(value));
}
pub fn calls() -> usize {
    CALLS.with(Cell::get)
}
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Plan {
    verify: bool,
    rows: bool,
    qmv: bool,
    matrix_affine: bool,
    matrix_packed: bool,
    qmv_address32: bool,
    batch_qmv: bool,
    streamed_qmv: bool,
    packed: bool,
    projections: [LinearMeta; 5],
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    kernel: i32,
    eps: u32,
}
pub(crate) fn forward(g: &Gdn, x: &Array, cache: &mut GdnCache) -> Result<Option<Array>> {
    // Compile only the fixed populated-cache paths. The reference owns cold
    // initialization/prefill and research GEMV/ops dispatch remains unchanged.
    if x.ndim() != 3
        || !(1..=8).contains(&x.shape()[0])
        || x.dtype() != Dtype::Bfloat16
        || !(x.shape()[1] == 1
            || (crate::verification::active() && (2..=8).contains(&x.shape()[1])))
        || crate::gemv_kernel::enabled()
        || std::env::var_os("RUST_MLX_GDN_OPS").is_some()
    {
        return Ok(None);
    }
    let (Some(conv), Some(state)) = (&cache.conv, &cache.state) else {
        return Ok(None);
    };
    let linears = [&g.qkv, &g.z, &g.a, &g.b, &g.out];
    if linears
        .iter()
        .any(|l| l.quant.as_ref().is_some_and(|q| q.mode != "affine"))
    {
        return Ok(None);
    }
    let p = Plan {
        verify: crate::verification::active(),
        rows: crate::verification::rows(),
        qmv: crate::qmv_kernel::enabled(),
        matrix_affine: crate::matrix_kernel::enabled(),
        matrix_packed: crate::matrix_kernel::packed(),
        qmv_address32: crate::qmv_kernel::address32(),
        batch_qmv: crate::qmv_kernel::batch_enabled(),
        streamed_qmv: crate::qmv_kernel::stream_x(),
        packed: g.packed_mode.get(),
        projections: linears.map(LinearMeta::from),
        hk: g.hk,
        hv: g.hv,
        dk: g.dk,
        dv: g.dv,
        kernel: g.kernel,
        eps: g.eps.to_bits(),
    };
    let mut args = vec![
        x.clone(),
        conv.clone(),
        state.clone(),
        g.conv.clone(),
        g.decode_conv.clone(),
        g.alog.clone(),
        g.dt.clone(),
        g.norm.clone(),
    ];
    for l in linears {
        append(l, &mut args);
    }
    let mut output = GRAPHS.with(|graphs| -> MlxResult<Vec<Array>> {
        let mut graphs = graphs.borrow_mut();
        let f = graphs.entry(p).or_insert_with(|| {
            Box::new(compile(
                move |args: &[Array]| -> MlxResult<Vec<Array>> {
                    let mut i = 8;
                    let g = Gdn {
                        packed_mode: Cell::new(p.packed),
                        qkv: p.projections[0].read(args, &mut i),
                        z: p.projections[1].read(args, &mut i),
                        a: p.projections[2].read(args, &mut i),
                        b: p.projections[3].read(args, &mut i),
                        out: p.projections[4].read(args, &mut i),
                        conv: args[3].clone(),
                        decode_conv: args[4].clone(),
                        alog: args[5].clone(),
                        dt: args[6].clone(),
                        norm: args[7].clone(),
                        hk: p.hk,
                        hv: p.hv,
                        dk: p.dk,
                        dv: p.dv,
                        kernel: p.kernel,
                        eps: f32::from_bits(p.eps),
                    };
                    let mut cache = GdnCache {
                        conv: Some(args[1].clone()),
                        state: Some(args[2].clone()),
                        ..Default::default()
                    };
                    let run = || -> Result<Vec<Array>> {
                        let y = g.forward_reference(&args[0], &mut cache)?;
                        let mut output = vec![
                            y,
                            cache.conv.context("compiled GDN convolution")?,
                            cache.state.context("compiled GDN state")?,
                        ];
                        if p.verify {
                            output.push(cache.verified_conv.context("compiled GDN conv history")?);
                            output.push(
                                cache
                                    .verified_states
                                    .context("compiled GDN state history")?,
                            );
                        }
                        Ok(output)
                    };
                    run().map_err(|e| Exception::custom(e.to_string()))
                },
                false,
            ))
        });
        f(&args)
    })?;
    ensure!(
        output.len() == if p.verify { 5 } else { 3 },
        "compiled GDN output arity"
    );
    if p.verify {
        cache.verified_states = output.pop();
        cache.verified_conv = output.pop();
    } else {
        cache.verified_states = None;
        cache.verified_conv = None;
    }
    cache.state = output.pop();
    cache.conv = output.pop();
    CALLS.with(|n| n.set(n.get().wrapping_add(1)));
    Ok(output.pop())
}
