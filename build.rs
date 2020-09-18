fn main() {
    if cfg!(feature = "enable_optix") {
        let optix_root =
            std::env::var("OPTIX_ROOT").expect("Could not get OPTIX_ROOT from environment");

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        let args = vec![format!("-I{}/include", optix_root), "-G".into()];

        compile_to_ptx("src/pathtracer/accelerator/device_programs.cu", &args);
    }
}

fn compile_to_ptx(cu_path: &str, args: &[String]) {
    println!("cargo:rerun-if-changed={}", cu_path);

    let full_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(cu_path);

    let mut ptx_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(cu_path);
    ptx_path.set_extension("ptx");
    std::fs::create_dir_all(ptx_path.parent().unwrap()).unwrap();

    let output = std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(&full_path)
        .arg("-o")
        .arg(&ptx_path)
        .args(args)
        .output()
        .expect("failed to fun nvcc");

    if !output.status.success() {
        panic!("{}", unsafe { String::from_utf8_unchecked(output.stderr) });
    }
}
