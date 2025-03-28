#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(feature = "cuda-from-host")]
mod cublasxt_from_host;
#[cfg(feature = "cuda-from-host")]
pub use cublasxt_from_host::*;

#[cfg(feature = "cuda-12040")]
mod cublasxt_12040;
#[cfg(feature = "cuda-12040")]
pub use cublasxt_12040::*;

#[cfg(feature = "cuda-12080")]
mod cublasxt_12080;
#[cfg(feature = "cuda-12080")]
pub use cublasxt_12080::*;
