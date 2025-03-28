use std::env;
use std::error;
use std::fs;
use std::path;
use std::time;

pub mod cuda_sdk;

const CUDA_BINDGEN_REGEN_ENV: &str = "CUDA_BINDGEN_REGEN";

fn main() {
    if env::var("DOCS_RS").is_ok() && cfg!(doc) {
        return;
    }
    let out_dir = path::PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR environment variable should be set by cargo."),
    );

    let sdk = cuda_sdk::CudaSdk::new().expect("Cannot create CUDA SDK instance.");
    // Emit metadata for the build script.
    println!("cargo::metadata=root={}", sdk.cuda_root().display());
    println!("cargo::metadata=version={}", sdk.cuda_version());
    println!("cargo::metadata=version_major={}", sdk.cuda_version_major());
    println!("cargo::metadata=version_minor={}", sdk.cuda_version_minor());
    // Re-run build script conditions.
    println!("cargo::rerun-if-changed=build/*.rs");
    for e in sdk.related_cuda_envs() {
        println!("cargo::rerun-if-env-changed={}", e);
    }

    let force_regen = match env::var(CUDA_BINDGEN_REGEN_ENV) {
        Ok(s) => s == "1",
        Err(_) => false,
    };
    if force_regen || cfg!(feature = "cuda-from-host") {
        let suffix = if force_regen {
            sdk.cuda_version().to_string()
        } else {
            "from_host".to_string()
        };
        maybe_create_cuda_driver_bindings(&sdk, force_regen, suffix.as_str());
        maybe_create_cublas_bindings(&sdk, force_regen, suffix.as_str());
        maybe_create_nvvm_bindings(&sdk, force_regen, suffix.as_str());
    }

    if cfg!(feature = "driver")
        || cfg!(feature = "cublas")
        || cfg!(feature = "cublaslt")
        || cfg!(feature = "cublasxt")
    {
        for libdir in sdk.cuda_library_paths() {
            println!("cargo::rustc-link-search=native={}", libdir.display());
        }
        println!("cargo::rustc-link-lib=dylib=cuda");
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
        fs::copy(sdk.libdevice_bitcode_path(), out_dir.join("libdevice.bc"))
            .expect("Cannot copy libdevice bitcode file.");
    }
}

fn maybe_create_cuda_driver_bindings(sdk: &cuda_sdk::CudaSdk, force_regen: bool, suffix: &str) {
    if !force_regen && !cfg!(feature = "driver") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("src/driver_sys/driver_{}.rs", suffix));
    let bindings = bindgen::Builder::default()
        .header("src/driver_sys/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.cuda_include_paths()
                .into_iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_type("^CU.*")
        .allowlist_type("^cuuint(32|64)_t")
        .allowlist_type("^cudaError_enum")
        .allowlist_type("^cu.*Complex$")
        .allowlist_type("^cuda.*")
        .allowlist_type("^libraryPropertyType.*")
        .allowlist_var("^CU.*")
        .allowlist_function("^cu.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .generate()
        .expect("Unable to generate CUDA driver bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write CUDA driver bindgen output to file.");
    reset_mtime(bindgen_path.as_path()).expect("Cannot reset mtime of CUDA driver bindings.");
}

fn maybe_create_cublas_bindings(sdk: &cuda_sdk::CudaSdk, force_regen: bool, suffix: &str) {
    let params = &[
        (
            cfg!(feature = "cublas") || force_regen,
            "cublas",
            "^cublas.*",
            "^CUBLAS.*",
        ),
        (
            cfg!(feature = "cublaslt") || force_regen,
            "cublaslt",
            "^cublasLt.*",
            "^CUBLASLT.*",
        ),
        (
            cfg!(feature = "cublasxt") || force_regen,
            "cublasxt",
            "^cublasXt.*",
            "^CUBLASXT.*",
        ),
    ];
    for (should_generate, pkg, tf, var) in params {
        if !should_generate {
            continue;
        }
        let bindgen_path = path::PathBuf::from(format!("src/{pkg}_sys/{pkg}_{suffix}.rs"));
        let bindings = bindgen::Builder::default()
            .header(format!("src/{pkg}_sys/wrapper.h"))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .clang_args(
                sdk.cuda_include_paths()
                    .into_iter()
                    .map(|p| format!("-I{}", p.display())),
            )
            .allowlist_type(tf)
            .allowlist_function(tf)
            .allowlist_var(var)
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: true,
            })
            .derive_default(true)
            .derive_eq(true)
            .derive_hash(true)
            .derive_ord(true)
            .size_t_is_usize(true)
            .layout_tests(true)
            .generate()
            .expect(format!("Unable to generate {pkg} bindings.").as_str());
        bindings
            .write_to_file(bindgen_path.as_path())
            .expect(format!("Cannot write {pkg} bindgen output to file.").as_str());
        reset_mtime(bindgen_path.as_path())
            .expect(format!("Cannot reset mtime of {pkg} bindings.").as_str());
    }
}

fn maybe_create_nvvm_bindings(sdk: &cuda_sdk::CudaSdk, force_regen: bool, suffix: &str) {
    if !force_regen && !cfg!(feature = "nvvm") {
        return;
    }
    let bindgen_path = path::PathBuf::from(format!("src/nvvm_sys/nvvm_{}.rs", suffix));
    let bindings = bindgen::Builder::default()
        .header("src/nvvm_sys/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(
            sdk.nvvm_include_paths()
                .into_iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("^nvvm.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .generate()
        .expect("Unable to generate libNVVM bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write libNVVM bindgen output to file.");
    reset_mtime(bindgen_path.as_path()).expect("Cannot reset mtime of libNVVM bindings.");
}

fn reset_mtime<P>(p: P) -> Result<(), Box<dyn error::Error>>
where
    P: AsRef<path::Path>,
{
    let f = fs::File::options().write(true).open(p.as_ref())?;
    f.set_modified(time::SystemTime::UNIX_EPOCH)?;
    Ok(())
}
