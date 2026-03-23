use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Re-run if the shader changes
    println!("cargo:rerun-if-changed=assets/shader.slang");

    let slangc = find_slangc();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let spv_out = out_dir.join("slang.spv");

    let status = Command::new(&slangc)
        .args([
            "assets/shader.slang",
            "-target",
            "spirv",
            "-profile",
            "spirv_1_4",
            "-emit-spirv-directly",
            "-fvk-use-entrypoint-name",
            "-entry",
            "vertMain",
            "-entry",
            "fragMain",
            "-o",
            spv_out.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("Failed to launch slangc at {slangc:?}: {e}"));

    assert!(status.success(), "slangc failed with {status}");
}

fn find_slangc() -> PathBuf {
    // 1. Explicit env override: SLANGC=/path/to/slangc cargo build
    if let Ok(path) = env::var("SLANGC") {
        return PathBuf::from(path);
    }

    // 2. Scan VULKAN_SDK (set by the LunarG SDK setup script)
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = PathBuf::from(sdk).join("bin").join("slangc");
        if candidate.exists() {
            return candidate;
        }
    }

    // 3. Fall back to PATH
    PathBuf::from("slangc")
}
