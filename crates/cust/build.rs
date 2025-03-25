use cust_raw::CUDA_VERSION;

fn main() {
    if CUDA_VERSION >= 12030 {
        println!("cargo:rustc-cfg=conditional_node");
    }
}
