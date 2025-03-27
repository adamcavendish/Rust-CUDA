use std::env;
use std::error;
use std::fs;
use std::path;
use std::time;

pub mod cuda_sdk;

const CUDA_BINDGEN_REGEN_ENV: &str = "CUDA_BINDGEN_REGEN";
const ENV_SEPARATOR: &str = if cfg!(target_os = "windows") {
    ";"
} else {
    ":"
};

fn main() {
    if env::var("DOCS_RS").is_ok() && cfg!(doc) {
        return;
    }

    let force_regen = match env::var(CUDA_BINDGEN_REGEN_ENV) {
        Ok(s) => s == "1",
        Err(_) => false,
    };

    let sdk = cuda_sdk::CudaSdk::new().expect("Cannot create CUDA SDK instance.");

    if force_regen || cfg!(feature = "cuda-from-host") {
        let suffix = if force_regen {
            sdk.cuda_version().to_string()
        } else {
            "from_host".to_string()
        };

        // CUDA Driver API Bindings.
        let bindgen_path = path::PathBuf::from(format!("src/driver/driver_{}.rs", suffix));
        let bindings =
            create_cuda_driver_bindings(&sdk).expect("Unable to generate CUDA driver bindings.");
        bindings
            .write_to_file(bindgen_path.as_path())
            .expect("Cannot write cuda_rt bindgen output to file.");
        reset_mtime(bindgen_path.as_path()).expect("Cannot reset mtime of CUDA driver bindings.");

        // cuBLAS API Bindings.
        let bindgen_path = path::PathBuf::from(format!("src/cublas/cublas_{}.rs", suffix));
        // @TODO/adamcavendish: finish cuBLAS.

        // libNVVM API Bindings.
        let bindgen_path = path::PathBuf::from(format!("src/nvvm/nvvm_{}.rs", suffix));
        // @TODO/adamcavendish: finish libNVVM.
    }

    // Emit metadata for the build script.
    println!("cargo::metadata=root={}", sdk.cuda_root().display());
    println!("cargo::metadata=version={}", sdk.cuda_version());
    println!("cargo::metadata=version_major={}", sdk.cuda_version_major());
    println!("cargo::metadata=version_minor={}", sdk.cuda_version_minor());

    println!("cargo::rerun-if-changed=build/*.rs");
    for libdir in sdk.cuda_library_paths() {
        println!("cargo::rustc-link-search=native={}", libdir.display());
    }
    println!("cargo::rustc-link-lib=dylib=cuda");
    for e in sdk.related_cuda_envs() {
        println!("cargo::rerun-if-env-changed={}", e);
    }
}

fn create_cuda_driver_bindings(
    sdk: &cuda_sdk::CudaSdk,
) -> Result<bindgen::Bindings, Box<dyn error::Error>> {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
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
        .generate()?;
    Ok(bindings)
}

fn reset_mtime<P>(p: P) -> Result<(), Box<dyn error::Error>>
where
    P: AsRef<path::Path>,
{
    let f = fs::File::options().write(true).open(p.as_ref())?;
    f.set_modified(time::SystemTime::UNIX_EPOCH)?;
    Ok(())
}
