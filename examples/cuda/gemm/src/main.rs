use std::error::Error;
use std::os;
use std::process;

use cuda_std::rt::sys as cuda_rt_sys;
use cust::memory::CopyDestination as _;
use cust::util::DeviceCopyExt as _;
use cust::util::SliceExt as _;
use ndarray::Array;
use ndarray_rand::RandomExt as _;
use ndarray_rand::rand_distr::Uniform;

const MAT_SIZES: [usize] = [128, 256, 512, 1024, 2048, 4096];
// static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;
    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    // let module = cust::module::Module::from_ptx(PTX, &[])?;
    // Make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None)?;

    // Make a cuBLAS context which manages the cuBLAS internal GPU memory allocations.
    let mut cublas_ctx = blastoff::CublasContext::new()?;

    // Create CUDA events to time the kernel execution.
    let beg_gpu = cuda_rt_sys::cudaEvent_t {};
    let beg_gpu = cuda_rt_sys::cudaEventCreateWithFlags();

    {
        let mat_a = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mat_b = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let mat_c_expect = ndarray::arr2(&[[23., 31.], [34., 46.]]);
        let (alpha, beta) = (1.0, 0.0);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mut mat_c_gpu = unsafe { cust::memory::DeviceBuffer::uninitialized(2 * 2)? };
        let alpha_gpu = cust::memory::DeviceBox::new(&alpha)?;
        let beta_gpu = cust::memory::DeviceBox::new(&beta)?;

        cublas_ctx.gemm::<f32>(
            &stream,
            2,
            2,
            2,
            &alpha_gpu,
            &mat_a_gpu,
            2,
            blastoff::MatrixOp::Transpose,
            &beta_gpu,
            &mat_b_gpu,
            2,
            blastoff::MatrixOp::Transpose,
            &mut mat_c_gpu,
            2,
        )?;

        let mut mat_c_actual = Array::<f32, _>::zeros((2, 2));
        mat_c_gpu.copy_to(&mut mat_c_actual.as_slice_mut().unwrap())?;
        assert_eq!(mat_c_actual, mat_c_expect);
    }

    for sz in MAT_SIZES.iter() {
        let mat_a = ndarray::arr2::<f32, _>::random((sz, sz), Uniform::new(-10., 10.));
        let mat_b = ndarray::arr2::<f32, _>::random((sz, sz), Uniform::new(-10., 10.));
        let mut mat_c = ndarray::arr2::<f32, _>::zeros((sz, sz));
        let (alpha, beta) = (1.0, 0.0);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_c_gpu = unsafe { cust::memory::DeviceBuffer::uninitialized(sz * sz)? };
        let alpha_gpu = cust::memory::DeviceBox::new(&alpha)?;
        let beta_gpu = cust::memory::DeviceBox::new(&beta)?;

        cublas_ctx.gemm::<f32>(
            &stream,
            2,
            2,
            2,
            &alpha_gpu,
            &mat_a_gpu,
            2,
            blastoff::MatrixOp::Transpose,
            &beta_gpu,
            &mat_b_gpu,
            2,
            blastoff::MatrixOp::Transpose,
            &mut mat_c_gpu,
            2,
        )?;
        mat_c_gpu.copy_to(&mut mat_c.as_slice_mut().unwrap())?;
        assert_eq!(mat_c, ndarray::arr2(&[[23., 31.], [34., 46.]]));
    }

    Ok(())
}
