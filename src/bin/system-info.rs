fn main() -> anyhow::Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&rust_mlx::environment::BenchmarkEnvironment::capture()?)?
    );
    Ok(())
}
