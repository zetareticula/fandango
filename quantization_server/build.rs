fn main() {
    // Simple build script - just set some basic environment variables
    println!("cargo:rustc-env=VERGEN_GIT_SHA=unknown");
    println!("cargo:rustc-env=VERGEN_BUILD_TIMESTAMP=unknown");
    println!("cargo:rerun-if-changed=build.rs");
}
