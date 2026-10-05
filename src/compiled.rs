//! Match the upstream compiled activation boundaries. Captured tensors are forbidden.
use mlx_rs::{Array, Dtype, error::Result, ops, transforms::compile::compile};
use std::cell::RefCell;
type Unary = Box<dyn FnMut(&Array) -> Result<Array>>;
type Binary = Box<dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array>>;
type Ternary = Box<dyn for<'a> FnMut((&'a Array, &'a Array, &'a Array)) -> Result<Array>>;
fn silu_graph(x: &Array) -> Result<Array> {
    x.multiply(ops::sigmoid(x)?)
}
fn swiglu_graph((gate, up): (&Array, &Array)) -> Result<Array> {
    silu_graph(gate)?.multiply(up)
}
fn decay_graph((alog, a, dt): (&Array, &Array, &Array)) -> Result<Array> {
    let x = a.add(dt)?;
    let softplus = ops::logaddexp(&x, Array::from_f32(0.).as_dtype(x.dtype())?)?;
    alog.as_dtype(Dtype::Float32)?
        .exp()?
        .negative()?
        .multiply(softplus)?
        .exp()
}
fn norm_graph((x, scale, eps): (&Array, &Array, &Array)) -> Result<Array> {
    let y = x.as_dtype(Dtype::Float32)?;
    let inv = y.square()?.mean_axis(-1, true)?.add(eps)?.rsqrt()?;
    y.multiply(inv)?
        .multiply(scale.as_dtype(Dtype::Float32)?.add(Array::from_f32(1.))?)?
        .as_dtype(x.dtype())
}
thread_local! {
    static SILU:RefCell<Unary>=RefCell::new(Box::new(compile(silu_graph,true)));
    static SWIGLU:RefCell<Binary>=RefCell::new(Box::new(compile(swiglu_graph,true)));
    static NORM:RefCell<Ternary>=RefCell::new(Box::new(compile(norm_graph,false)));
    static DECAY:RefCell<Ternary>=RefCell::new(Box::new(compile(decay_graph,true)));
}
pub fn silu(x: &Array) -> Result<Array> {
    SILU.with(|f| f.borrow_mut()(x))
}
pub fn swiglu(gate: &Array, up: &Array) -> Result<Array> {
    SWIGLU.with(|f| f.borrow_mut()((gate, up)))
}
pub fn decay(alog: &Array, a: &Array, dt: &Array) -> Result<Array> {
    DECAY.with(|f| f.borrow_mut()((alog, a, dt)))
}

pub fn norm(x: &Array, scale: &Array, eps: f32) -> Result<Array> {
    NORM.with(|f| f.borrow_mut()((x, scale, &Array::from_f32(eps))))
}

fn activate_graph((x, count): (&Array, &Array)) -> Result<Array> {
    silu_graph(&x.divide(count)?)
}
fn injection_graph((x, count): (&Array, &Array)) -> Result<Array> {
    ops::sigmoid(x.divide(count)?)?.multiply(Array::from_f32(2.).as_dtype(x.dtype())?)
}
thread_local! {
 static ACTIVATE:RefCell<Binary>=RefCell::new(Box::new(compile(activate_graph,true)));
 static INJECTION:RefCell<Binary>=RefCell::new(Box::new(compile(injection_graph,true)));
}
pub fn activate(x: &Array, count: i32) -> Result<Array> {
    ACTIVATE.with(|f| f.borrow_mut()((x, &Array::from_f32(count as f32).as_dtype(x.dtype())?)))
}
pub fn injection(x: &Array, count: i32) -> Result<Array> {
    INJECTION.with(|f| f.borrow_mut()((x, &Array::from_f32(count as f32).as_dtype(x.dtype())?)))
}
thread_local! {
 static MIX:RefCell<std::collections::HashMap<i32,Binary>>=RefCell::new(std::collections::HashMap::new());
}
pub fn mix(projected: &Array, normed: &Array, count: i32) -> Result<Array> {
    MIX.with(|cell| {
        let mut cell = cell.borrow_mut();
        let f = cell.entry(count).or_insert_with(|| {
            Box::new(compile(
                move |(p, n): (&Array, &Array)| {
                    let s = n.shape();
                    let shape = [s[0], s[1], count, s[2] / count];
                    ops::sigmoid(p)?
                        .reshape(&shape)?
                        .multiply(n.reshape(&shape)?)?
                        .mean_axis(-2, false)
                },
                false,
            ))
        });
        f((projected, normed))
    })
}

fn decode_conv_graph((x, w): (&Array, &Array)) -> Result<Array> {
    x.as_dtype(Dtype::Float32)?
        .multiply(w.expand_dims(0)?)?
        .sum_axis(1, false)?
        .as_dtype(x.dtype())?
        .expand_dims(1)
}
thread_local! {static DECODE_CONV:RefCell<Binary>=RefCell::new(Box::new(compile(decode_conv_graph,true)));}
pub fn decode_conv(x: &Array, w: &Array) -> Result<Array> {
    DECODE_CONV.with(|f| f.borrow_mut()((x, w)))
}
