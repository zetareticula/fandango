use std::env;
use std::path::PathBuf;

fn main() {
    // Generate the default 'cargo:' instruction output
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    
    // Generate version information using vergen
    let mut config = vergen::Config::default();
    *config.git_mut().sha_kind_mut() = vergen::ShaKind::Short;
    *config.git_mut().commit_timestamp_kind_mut() = vergen::TimestampKind::DateOnly;
    
    // Generate the version information
    if let Err(e) = vergen::vergen(config) {
        eprintln!("Failed to generate version information: {}", e);
        std::process::exit(1);
    }
    
    // Rebuild if the build script changes
    println!("cargo:rerun-if-changed=build.rs");
}
