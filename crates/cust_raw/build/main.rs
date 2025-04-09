//! # The build script for cust_raw
//! The build script for the cust_raw generates bindings for libraries in the
//! CUDA SDK. The build scripts searches for the CUDA SDK by reading the
//! `CUDA_PATH`, `CUDA_ROOT`, or `CUDA_TOOLKIT_ROOT_DIR` environment variables
//! in that order. If none of these variables are set to a vaild CUDA Toolkit
//! SDK path, the build script will attempt to search for any SDK in the
//! default installation locations for the current platform.
//!
//! ## Bindings
//! You can control which bindings are generated by enabling features in your
//! `Cargo.toml` file. By default, only the CUDA driver API is enabled.
//!
//! ## Cargo metadata
//! The build script emits Cargo metadata that can be used by dependent crates
//! in their build script. You can read this metadata via `DEP_CUDA_*`
//! environment variables. The current list of metadata includes:
//!
//! - `DEP_CUDA_ROOT`: The root directory of the CUDA SDK installation used.
//! - `DEP_CUDA_DRIVER_VERSION`: The version of the CUDA driver API found (e.g: `12080`).
//! - `DEP_CUDA_DRIVER_VERSION_MAJOR`: The major version of the CUDA driver API found.
//! - `DEP_CUDA_DRIVER_VERSION_MINOR`: The minor version of the CUDA driver API found.
//! - `DEP_CUDA_RUNTIME_VERSION`: The version of the CUDA runtime API found.
//! - `DEP_CUDA_INCLUDES`: The include directories for the CUDA SDK, separated by platform-specific path separator.
//! - `DEP_CUDA_NVVM_INCLUDES`: The include directories for NVVM headers, separated by platform-specific path separator.
//!

use std::env;
use std::fs;
use std::path;

pub mod callbacks;
pub mod cuda_sdk;

fn main() {
    let outdir = path::PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR environment variable should be set by cargo."),
    );

    let sdk = cuda_sdk::CudaSdk::new().expect("Cannot create CUDA SDK instance.");
    // Emit metadata for the build script.
    println!("cargo::metadata=root={}", sdk.cuda_root().display());
    println!("cargo::metadata=driver_version={}", sdk.driver_version());
    println!(
        "cargo::metadata=driver_version_major={}",
        sdk.driver_version_major()
    );
    println!(
        "cargo::metadata=driver_version_minor={}",
        sdk.driver_version_minor()
    );
    println!("cargo::metadata=runtime_version={}", sdk.runtime_version());
    let metadata_cuda_include = env::join_paths(sdk.cuda_include_paths())
        .map(|s| s.to_string_lossy().to_string())
        .expect("Failed to build metadata for cuda_include.");
    let metadata_nvvm_include = env::join_paths(sdk.nvvm_include_paths())
        .map(|s| s.to_string_lossy().to_string())
        .expect("Failed to build metadata for nvvm_include.");
    println!("cargo::metadata=includes={}", metadata_cuda_include);
    println!("cargo::metadata=nvvm_includes={}", metadata_nvvm_include);
    // Re-run build script conditions.
    println!("cargo::rerun-if-changed=build");
    for e in sdk.related_cuda_envs() {
        println!("cargo::rerun-if-env-changed={}", e);
    }

    create_cuda_driver_bindings(&sdk, outdir.as_path());
    create_cuda_runtime_bindings(&sdk, outdir.as_path());
    create_cublas_bindings(&sdk, outdir.as_path());
    create_nptx_compiler_bindings(&sdk, outdir.as_path());
    create_nvvm_bindings(&sdk, outdir.as_path());

    if cfg!(any(
        feature = "driver",
        feature = "runtime",
        feature = "cublas",
        feature = "cublaslt",
        feature = "cublasxt"
    )) {
        for libdir in sdk.cuda_library_paths() {
            println!("cargo::rustc-link-search=native={}", libdir.display());
        }
        println!("cargo::rustc-link-lib=dylib=cuda");
    }
    if cfg!(feature = "runtime") {
        println!("cargo::rustc-link-lib=dylib=cudart");
    }
    if cfg!(feature = "cublas") || cfg!(feature = "cublasxt") {
        println!("cargo::rustc-link-lib=dylib=cublas");
    }
    if cfg!(feature = "cublaslt") {
        println!("cargo::rustc-link-lib=dylib=cublaslt");
    }
    if cfg!(feature = "nvvm") {
        for libdir in sdk.nvvm_library_paths() {
            println!("cargo::rustc-link-search=native={}", libdir.display());
        }
        println!("cargo::rustc-link-lib=dylib=nvvm");
        // Handle libdevice support.
        fs::copy(sdk.libdevice_bitcode_path(), outdir.join("libdevice.bc"))
            .expect("Cannot copy libdevice bitcode file.");
    }
}

fn create_cuda_driver_bindings(sdk: &cuda_sdk::CudaSdk, outdir: &path::Path) {
    if !cfg!(feature = "driver") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("{}/driver_sys.rs", outdir.display()));
    let header = "build/driver_wrapper.h";
    let bindings = bindgen::Builder::default()
        .header(header)
        .parse_callbacks(Box::new(callbacks::FunctionRenames::new(
            "cu",
            outdir,
            header,
            sdk.cuda_include_paths().to_owned(),
        )))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.cuda_include_paths()
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_type("^CU.*")
        .allowlist_type("^cuuint(32|64)_t")
        .allowlist_type("^cudaError_enum")
        .allowlist_type("^cu.*Complex$")
        .allowlist_type("^cuda.*")
        .allowlist_var("^CU.*")
        .allowlist_function("^cu.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .must_use_type("CUresult")
        .generate()
        .expect("Unable to generate CUDA driver bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write CUDA driver bindgen output to file.");
}

fn create_cuda_runtime_bindings(sdk: &cuda_sdk::CudaSdk, outdir: &path::Path) {
    if !cfg!(feature = "runtime") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("{}/runtime_sys.rs", outdir.display()));
    let header = "build/runtime_wrapper.h";
    let bindings = bindgen::Builder::default()
        .header(header)
        .parse_callbacks(Box::new(callbacks::FunctionRenames::new(
            "cuda",
            outdir,
            header,
            sdk.cuda_include_paths().to_owned(),
        )))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.cuda_include_paths()
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_type("^CU.*")
        .allowlist_type("^cuda.*")
        .allowlist_type("^libraryPropertyType.*")
        .allowlist_var("^CU.*")
        .allowlist_function("^cu.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .must_use_type("cudaError_t")
        .generate()
        .expect("Unable to generate CUDA runtime bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write CUDA runtime bindgen output to file.");
}

fn create_cublas_bindings(sdk: &cuda_sdk::CudaSdk, outdir: &path::Path) {
    #[rustfmt::skip]
    let params = &[
        (cfg!(feature = "cublas"), "cublas", "^cublas.*", "^CUBLAS.*"),
        (cfg!(feature = "cublaslt"), "cublasLt", "^cublasLt.*", "^CUBLASLT.*"),
        (cfg!(feature = "cublasxt"), "cublasXt", "^cublasXt.*", "^CUBLASXT.*"),
    ];
    for (should_generate, pkg, tf, var) in params {
        if !should_generate {
            continue;
        }
        let bindgen_path = path::PathBuf::from(format!("{}/{pkg}_sys.rs", outdir.display()));
        let header = format!("build/{pkg}_wrapper.h");
        let bindings = bindgen::Builder::default()
            .header(&header)
            .parse_callbacks(Box::new(callbacks::FunctionRenames::new(
                pkg,
                outdir,
                header,
                sdk.cuda_include_paths().to_owned(),
            )))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .clang_args(
                sdk.cuda_include_paths()
                    .iter()
                    .map(|p| format!("-I{}", p.display())),
            )
            .allowlist_type(tf)
            .allowlist_function(tf)
            .allowlist_var(var)
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(true)
            .derive_eq(true)
            .derive_hash(true)
            .derive_ord(true)
            .size_t_is_usize(true)
            .layout_tests(true)
            .must_use_type("cublasStatus_t")
            .generate()
            .unwrap_or_else(|_| panic!("Unable to generate {pkg} bindings."));
        bindings
            .write_to_file(bindgen_path.as_path())
            .unwrap_or_else(|_| panic!("Cannot write {pkg} bindgen output to file."));
    }
}

fn create_nptx_compiler_bindings(sdk: &cuda_sdk::CudaSdk, outdir: &path::Path) {
    if !cfg!(feature = "nvptx-compiler") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("{}/nvptx_compiler_sys.rs", outdir.display()));
    let bindings = bindgen::Builder::default()
        .header("build/nvptx_compiler_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.cuda_include_paths()
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("^nvPTX.*")
        .allowlist_type("^nvPTX.*")
        .allowlist_var("^NVPTX.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .must_use_type("nvPTXCompileResult")
        .generate()
        .expect("Unable to generate nvptx-compiler bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write nvptx-compiler bindgen output to file.");
}

fn create_nvvm_bindings(sdk: &cuda_sdk::CudaSdk, outdir: &path::Path) {
    if !cfg!(feature = "nvvm") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("{}/nvvm_sys.rs", outdir.display()));
    let bindings = bindgen::Builder::default()
        .header("build/nvvm_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.nvvm_include_paths()
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("^nvvm.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .must_use_type("nvvmResult")
        .generate()
        .expect("Unable to generate libNVVM bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write libNVVM bindgen output to file.");
}
