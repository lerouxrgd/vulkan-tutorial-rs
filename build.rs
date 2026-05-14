use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/shaders/scene.slang");
    println!("cargo:rerun-if-changed=assets/shaders/particles.slang");

    let slangc = find_slangc();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    compile_shader(
        &slangc,
        "assets/shaders/scene.slang",
        out_dir.join("scene.spv"),
        &["-entry", "vertMain", "-entry", "fragMain"],
    );

    compile_shader(
        &slangc,
        "assets/shaders/particles.slang",
        out_dir.join("particles_graphics.spv"),
        &["-entry", "vertMain", "-entry", "fragMain"],
    );

    compile_shader(
        &slangc,
        "assets/shaders/particles.slang",
        out_dir.join("particles_compute.spv"),
        &["-entry", "compMain"],
    );
}

fn compile_shader(slangc: &PathBuf, src: &str, out: PathBuf, entries: &[&str]) {
    let mut cmd = Command::new(slangc);
    cmd.args([
        src,
        "-target",
        "spirv",
        "-profile",
        "spirv_1_4",
        "-emit-spirv-directly",
        "-fvk-use-entrypoint-name",
    ]);
    cmd.args(entries);
    cmd.args(["-o", out.to_str().unwrap()]);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to launch slangc at {slangc:?}: {e}"));

    assert!(status.success(), "slangc failed on {src} with {status}");
}

fn find_slangc() -> PathBuf {
    if let Ok(path) = env::var("SLANGC") {
        return PathBuf::from(path);
    }

    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = PathBuf::from(sdk).join("bin").join("slangc");
        if candidate.exists() {
            return candidate;
        }
    }

    PathBuf::from("slangc")
}
