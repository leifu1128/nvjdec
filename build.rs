use std::{env, fs};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

fn gen_nv_bindings() {
    let clang_args = vec![format!("-I{}/include", env::var("CUDA_PATH_V11_8").unwrap())];
    let wrapper_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("nv_wrapper.h");
    let wrapper_path = wrapper_path.to_str().unwrap();
    let _ = fs::remove_file(wrapper_path);
    let mut wrapper = File::create(wrapper_path).unwrap();
    writeln!(wrapper, "#include <cuda_runtime_api.h>").unwrap();
    writeln!(wrapper, "#include <nvjpeg.h>").unwrap();

    let allows = "#![allow(dead_code)]\n#![allow(non_upper_case_globals)]\n#![allow(non_camel_case_types)]\n#![allow(non_snake_case)]";
    let bindings = bindgen::Builder::default()
        .header(wrapper_path)
        .raw_line(allows)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .anon_fields_prefix("un")
        .derive_debug(true)
        .impl_debug(false)
        .derive_default(true)
        .derive_copy(false)
        .impl_partialeq(true)
        .merge_extern_blocks(true)
        .generate_comments(false)
        .allowlist_function("^cuda.*")
        .allowlist_function("^nvjpeg.*")
        .allowlist_var("^NVJPEG.*")
        .clang_args(&clang_args)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/nv file.
    let _ = fs::remove_file("src/nv.rs");

    bindings
        .write_to_file("src/nv.rs")
        .expect("Couldn't write bindings!");

    let mut target_file = OpenOptions::new().append(true).open("src/nv.rs").unwrap();
    let mut source_file = fs::File::open("src/nv_custom.rs").unwrap();
    let mut content = Vec::new();

    source_file.read_to_end(&mut content).unwrap();
    target_file.write_all(&content).unwrap();
}

fn main() {
    println!("cargo:rerun-if-changed=src/decoder.rs");
    println!("cargo:rerun-if-changed=src/nv_custom.rs");

    println!("cargo:rustc-link-search={}", env::var("CUDA_PATH").unwrap());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvjpeg");
}