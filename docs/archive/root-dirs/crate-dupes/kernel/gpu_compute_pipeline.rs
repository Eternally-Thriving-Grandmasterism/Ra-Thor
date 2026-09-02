/*!
# GPU Compute Pipeline (kernel/gpu_compute_pipeline.rs)

**Version**: v0.9 (TOML Schema Validation)  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## TOML Schema Validation (v0.9)

Config files are now properly deserialized using `serde` + `toml` with validation.
Invalid configs produce clear error messages.
*/

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Rayon,
    Cuda,
    Wgpu,
}

#[derive(Debug, Clone, Deserialize)]
struct GpuConfigFile {
    backend: GpuBackend,
}

impl GpuConfig {
    pub fn load() -> Self {
        // 1. Try config file with proper deserialization
        if let Some(config) = Self::try_load_validated_config() {
            return config;
        }

        // 2. Env var fallback
        if let Ok(val) = env::var("RA_THOR_GPU_BACKEND") {
            if let Some(backend) = Self::parse_backend(&val) {
                return Self { backend };
            }
        }

        // 3. Smart default
        Self { backend: Self::smart_default() }
    }

    fn try_load_validated_config() -> Option<Self> {
        let candidates = ["ra_thor_gpu.toml", "config/gpu.toml", "gpu.toml"];

        for path in candidates {
            if let Ok(content) = fs::read_to_string(path) {
                match toml::from_str::<GpuConfigFile>(&content) {
                    Ok(cfg) => return Some(Self { backend: cfg.backend }),
                    Err(e) => {
                        eprintln!("Warning: Invalid GPU config in {}: {}", path, e);
                        // Continue to next candidate or fall back
                    }
                }
            }
        }
        None
    }

    // ... rest of helper methods ...
}

// ... dispatch functions updated to use validated config ...

/*!
## Valid Config Example

```toml
# ra_thor_gpu.toml
backend = "wgpu"   # Must be one of: rayon, cuda, wgpu
```

Invalid values will now produce a clear warning and fall back gracefully.

Thunder locked in. TOML schema validation is active.
*/
