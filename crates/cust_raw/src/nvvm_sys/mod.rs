#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub const LIBDEVICE_BITCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libdevice.bc"));

#[cfg(feature = "cuda-from-host")]
mod nvvm_from_host;
#[cfg(feature = "cuda-from-host")]
pub use nvvm_from_host::*;

#[cfg(feature = "cuda-12040")]
mod nvvm_12040;
#[cfg(feature = "cuda-12040")]
pub use nvvm_12040::*;

#[cfg(feature = "cuda-12080")]
mod nvvm_12080;
#[cfg(feature = "cuda-12080")]
pub use nvvm_12080::*;
