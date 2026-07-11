/*!
# GPU Compute Pipeline (kernel/gpu_compute_pipeline.rs)

**Version**: v0.8 (Config File Backend Selection)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Config File Backend Selection (v0.8)

Backend can now be selected via (in priority order):

1. **Config file** (`ra_thor_gpu.toml` or `config/gpu.toml`)
2. **Environment variable** `RA_THOR_GPU_BACKEND`
3. **Smart default** based on compiled features

Example `ra_thor_gpu.toml`:
```toml
backend = "wgpu"   # or "cuda" or "rayon"
```
*/

use std::env;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GpuBackend {
    #[default]
    Rayon,
    Cuda,
    Wgpu,
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub backend: GpuBackend,
}

impl GpuConfig {
    /// Load config from file, env var, or smart default (in that order).
    pub fn load() -> Self {
        // 1. Try config file
        if let Some(backend) = Self::try_load_from_file() {
            return Self { backend };
        }

        // 2. Try environment variable
        if let Ok(val) = env::var("RA_THOR_GPU_BACKEND") {
            if let Some(backend) = Self::parse_backend(&val) {
                return Self { backend };
            }
        }

        // 3. Smart default based on compiled features
        Self { backend: Self::smart_default() }
    }

    fn try_load_from_file() -> Option<GpuBackend> {
        let candidates = ["ra_thor_gpu.toml", "config/gpu.toml", "gpu.toml"];

        for path in candidates {
            if let Ok(content) = fs::read_to_string(path) {
                if let Some(backend) = Self::parse_toml_backend(&content) {
                    return Some(backend);
                }
            }
        }
        None
    }

    fn parse_toml_backend(content: &str) -> Option<GpuBackend> {
        // Very simple TOML parser for "backend = \"wgpu\"" style
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("backend") {
                if let Some(val) = line.split('=').nth(1) {
                    let val = val.trim().trim_matches('"').trim_matches('\'');
                    return Self::parse_backend(val);
                }
            }
        }
        None
    }

    fn parse_backend(s: &str) -> Option<GpuBackend> {
        match s.to_lowercase().as_str() {
            "wgpu"  => Some(GpuBackend::Wgpu),
            "cuda"  => Some(GpuBackend::Cuda),
            "rayon" => Some(GpuBackend::Rayon),
            _ => None,
        }
    }

    fn smart_default() -> GpuBackend {
        #[cfg(feature = "wgpu")] { return GpuBackend::Wgpu; }
        #[cfg(all(feature = "cuda", not(feature = "wgpu")))] { return GpuBackend::Cuda; }
        GpuBackend::Rayon
    }
}

// Update the dispatch functions to use GpuConfig::load()
pub fn gpu_deliberation_batch(...) -> Vec<(String, f64, f64)> {
    let config = GpuConfig::load();

    match config.backend {
        GpuBackend::Wgpu => { /* wgpu path */ }
        GpuBackend::Cuda => { /* cuda path */ }
        GpuBackend::Rayon => { /* rayon path */ }
    }
}

// ... rest of file ...

/*!
## Config File Example

Create `ra_thor_gpu.toml` in the working directory:

```toml
backend = "wgpu"
```

Or place it in `config/gpu.toml`.

Thunder locked in. Config file backend selection is now supported.
*/
