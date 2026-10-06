//! The only custom-kernel FFI boundary. MLX retains lazy graph inputs and owns GPU synchronization.
use anyhow::{Result, ensure};
use mlx_rs::{Array, Dtype, Stream, ops};
use mlx_sys as sys;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{ffi::CString, marker::PhantomData, rc::Rc};

static CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Lazy views have provisional strides: only an available array has its final layout.
pub(crate) fn is_evaluated_row_contiguous(array: &Array) -> Result<bool> {
    let mut available = false;
    // SAFETY: the borrowed Array handle stays live and the bool out-pointer is
    // initialized stack storage. This query neither evaluates nor reads GPU data.
    check(
        unsafe { sys::_mlx_array_is_available(&mut available, array.as_ptr()) },
        "query available",
    )?;
    if !available {
        return Ok(false);
    }
    let mut contiguous = false;
    // SAFETY: same live borrowed handle and valid bool storage; availability
    // above ensures layout flags describe the evaluated buffer, not a lazy view.
    check(
        unsafe { sys::_mlx_array_is_row_contiguous(&mut contiguous, array.as_ptr()) },
        "query row contiguous",
    )?;
    Ok(contiguous)
}
/// Process-exclusive capture, stopped on this owning thread even after errors.
pub struct Capture {
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Capture {
    pub fn start(path: &std::path::Path) -> Result<Self> {
        ensure!(!path.exists(), "capture output already exists");
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("capture path must be UTF-8"))?,
        )?;
        let _ = ops::zeros_dtype(&[1], Dtype::Float32)?;
        ensure!(
            CAPTURE_ACTIVE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "capture already active"
        );
        // SAFETY: the CString remains live for the synchronous call; MLX copies
        // its path. The atomic reservation excludes any second capture owner.
        let status = unsafe { sys::mlx_metal_start_capture(path.as_ptr()) };
        if status != 0 {
            CAPTURE_ACTIVE.store(false, Ordering::Release);
            check(status, "start capture")?;
        }
        Ok(Self {
            active: true,
            _thread_bound: PhantomData,
        })
    }
    pub fn finish(mut self) -> Result<()> {
        self.stop()
    }
    fn stop(&mut self) -> Result<()> {
        if self.active {
            // SAFETY: this thread owns the sole active process capture. MLX's
            // synchronous stop consumes no borrowed pointers or array handles.
            let status = unsafe { sys::mlx_metal_stop_capture() };
            self.active = false;
            CAPTURE_ACTIVE.store(false, Ordering::Release);
            check(status, "stop capture")?;
        }
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn check(status: i32, operation: &str) -> Result<()> {
    ensure!(status == 0, "MLX C API {operation} returned {status}");
    Ok(())
}
struct Strings(sys::mlx_vector_string);
impl Strings {
    fn new(values: &[&str]) -> Result<Self> {
        let names = values
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut ptrs = names.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        // SAFETY: names and pointer array stay alive during the call. MLX copies strings.
        let h = unsafe { sys::mlx_vector_string_new_data(ptrs.as_mut_ptr(), ptrs.len()) };
        ensure!(!h.ctx.is_null(), "cannot allocate kernel names");
        Ok(Self(h))
    }
}
impl Drop for Strings {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns exactly one independent vector, freed once here.
        unsafe {
            sys::mlx_vector_string_free(self.0);
        }
    }
}
struct Arrays(sys::mlx_vector_array);
impl Arrays {
    fn new(inputs: &[&Array]) -> Result<Self> {
        let ptrs = inputs.iter().map(|a| a.as_ptr()).collect::<Vec<_>>();
        // SAFETY: live Array handles are borrowed for the call; MLX copies/retains their values.
        let h = unsafe { sys::mlx_vector_array_new_data(ptrs.as_ptr(), ptrs.len()) };
        ensure!(!h.ctx.is_null(), "cannot allocate array vector");
        Ok(Self(h))
    }
    fn empty() -> Result<Self> {
        // SAFETY: no input pointer; returned handle is uniquely owned by this wrapper.
        let h = unsafe { sys::mlx_vector_array_new() };
        ensure!(!h.ctx.is_null(), "cannot allocate output vector");
        Ok(Self(h))
    }
    fn get(&self, index: usize) -> Result<Array> {
        // SAFETY: new() creates an independent empty handle; get copies a retained value
        // into it. On failure it is freed; on success ownership transfers to Array.
        unsafe {
            let mut h = sys::mlx_array_new();
            let status = sys::mlx_vector_array_get(&mut h, self.0, index);
            if status != 0 {
                sys::mlx_array_free(h);
                check(status, "get output")?;
            }
            Ok(Array::from_ptr(h))
        }
    }
}
impl Drop for Arrays {
    fn drop(&mut self) {
        // SAFETY: wrapper owns this vector; returned arrays have independent retained handles.
        unsafe {
            sys::mlx_vector_array_free(self.0);
        }
    }
}
struct Config(sys::mlx_fast_metal_kernel_config);
impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: unique config handle, freed after apply has copied its settings.
        unsafe {
            sys::mlx_fast_metal_kernel_config_free(self.0);
        }
    }
}
pub enum Template<'a> {
    Int(&'a str, i32),
    Dtype(&'a str, Dtype),
    Bool(&'a str, bool),
}
pub struct Kernel {
    handle: sys::mlx_fast_metal_kernel,
    inputs: usize,
    outputs: usize,
    _thread_bound: PhantomData<Rc<()>>,
}
pub struct Launch<'a> {
    pub inputs: &'a [&'a Array],
    pub templates: &'a [Template<'a>],
    pub outputs: &'a [(&'a [i32], Dtype)],
    pub grid: [i32; 3],
    pub group: [i32; 3],
}
impl Kernel {
    pub fn new(name: &str, inputs: &[&str], outputs: &[&str], source: &str) -> Result<Self> {
        Self::with_header(name, inputs, outputs, source, "")
    }
    pub fn with_header(
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        source: &str,
        header: &str,
    ) -> Result<Self> {
        // Install mlx-rs's non-aborting error handler through a safe operation first.
        let _ = ops::zeros_dtype(&[1], Dtype::Float32)?;
        let name = CString::new(name)?;
        let source = CString::new(source)?;
        let header = CString::new(header)?;
        let inputs_h = Strings::new(inputs)?;
        let outputs_h = Strings::new(outputs)?;
        // SAFETY: all C strings/vectors live through construction and MLX copies them.
        // The returned kernel is owned by Kernel and cannot move between threads.
        let handle = unsafe {
            sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                inputs_h.0,
                outputs_h.0,
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        };
        ensure!(!handle.ctx.is_null(), "cannot create Metal kernel");
        Ok(Self {
            handle,
            inputs: inputs.len(),
            outputs: outputs.len(),
            _thread_bound: PhantomData,
        })
    }
    pub fn launch(&self, launch: Launch<'_>) -> Result<Vec<Array>> {
        ensure!(
            launch.inputs.len() == self.inputs && launch.outputs.len() == self.outputs,
            "kernel arity mismatch"
        );
        ensure!(
            launch.grid.iter().chain(&launch.group).all(|&v| v > 0),
            "invalid kernel grid"
        );
        ensure!(
            launch.group.iter().map(|&n| n as i64).product::<i64>() <= 1024,
            "oversized threadgroup"
        );
        ensure!(
            launch
                .outputs
                .iter()
                .all(|(s, _)| !s.is_empty() && s.iter().all(|&d| d > 0)),
            "invalid output shape"
        );
        let inputs = Arrays::new(launch.inputs)?;
        let mut outputs = Arrays::empty()?;
        // SAFETY: unique config handle; setters copy shapes/templates while CString lives.
        // Input arrays/stream remain alive through apply; MLX retains them in its lazy
        // graph. Shapes, group and arity are validated above; specialized callers
        // validate their own buffer indexing contract. MLX synchronizes evaluation.
        unsafe {
            let config = Config(sys::mlx_fast_metal_kernel_config_new());
            ensure!(!config.0.ctx.is_null(), "cannot allocate kernel config");
            for &(shape, dtype) in launch.outputs {
                check(
                    sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config.0,
                        shape.as_ptr(),
                        shape.len(),
                        dtype.into(),
                    ),
                    "output shape",
                )?;
            }
            for template in launch.templates {
                let status = match template {
                    Template::Int(name, value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config.0,
                            CString::new(*name)?.as_ptr(),
                            *value,
                        )
                    }
                    Template::Dtype(name, value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config.0,
                            CString::new(*name)?.as_ptr(),
                            (*value).into(),
                        )
                    }
                    Template::Bool(name, value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config.0,
                            CString::new(*name)?.as_ptr(),
                            *value,
                        )
                    }
                };
                check(status, "template")?;
            }
            let [x, y, z] = launch.grid;
            check(
                sys::mlx_fast_metal_kernel_config_set_grid(config.0, x, y, z),
                "grid",
            )?;
            let [x, y, z] = launch.group;
            check(
                sys::mlx_fast_metal_kernel_config_set_thread_group(config.0, x, y, z),
                "threadgroup",
            )?;
            let stream = Stream::thread_local_or_default();
            check(
                sys::mlx_fast_metal_kernel_apply(
                    &mut outputs.0,
                    self.handle,
                    inputs.0,
                    config.0,
                    stream.as_ptr(),
                ),
                "kernel apply",
            )?;
        }
        (0..self.outputs).map(|i| outputs.get(i)).collect()
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: sole kernel handle owner. Lazy MLX graphs retain their primitive separately.
        unsafe {
            sys::mlx_fast_metal_kernel_free(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn custom_kernel_roundtrip_and_owned_output() {
        let x = Array::from_slice(&[1f32, -2., 3., 0., 0.5], &[5]);
        let k = Kernel::new(
            "rust_mlx_square",
            &["x"],
            &["y"],
            "uint i=thread_position_in_grid.x; if (i < N) y[i]=x[i]*x[i];",
        )
        .unwrap();
        let mut y = k
            .launch(Launch {
                inputs: &[&x],
                templates: &[Template::Int("N", 5)],
                outputs: &[(&[5], Dtype::Float32)],
                grid: [32, 1, 1],
                group: [32, 1, 1],
            })
            .unwrap();
        drop(k);
        drop(x);
        let y = y.remove(0);
        y.eval().unwrap();
        assert_eq!(y.as_slice::<f32>(), &[1., 4., 9., 0., 0.25]);
    }
}
