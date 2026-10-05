use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};
use std::process::Command;

fn command(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".into())
}
#[derive(Serialize)]
pub struct BenchmarkEnvironment {
    pub hardware: Value,
    pub software: Value,
    pub revision: Value,
    pub captured_unix_seconds: u64,
}
impl BenchmarkEnvironment {
    pub fn capture() -> Result<Self> {
        let raw = command(
            "system_profiler",
            &["SPHardwareDataType", "SPDisplaysDataType", "-json"],
        );
        let profile: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
        let h = &profile["SPHardwareDataType"][0];
        let g = &profile["SPDisplaysDataType"][0];
        Ok(Self {
            hardware: json!({"model":h["machine_model"],"chip":h["chip_type"],"memory":h["physical_memory"],"gpu":g["sppci_model"],"gpu_cores":g["sppci_cores"]}),
            software: json!({"macos":command("sw_vers", &["-productVersion"]),"macos_build":command("sw_vers", &["-buildVersion"]),"xcode":command("xcodebuild", &["-version"]),"rust":command("rustc", &["--version"]),"mlx_rs":"0.32.0","mlx_sys":"0.6.0","mlx":"0.32.2","package_version":env!("CARGO_PKG_VERSION")}),
            revision: json!({"commit":command("git", &["rev-parse","HEAD"]),"dirty":!command("git", &["status","--porcelain"]).is_empty()}),
            captured_unix_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        })
    }
}
