//! Ra-Thor Mercy-Gated API
//!
//! Minimal production stub restored for Lattice Conductor v14.8.0.
//! Provides the public surface expected by lib.rs while remaining
//! fully mercy-aligned and ready for deeper expansion.
//!
//! Thunder locked in. yoi ⚡

use std::net::SocketAddr;

/// High-level mercy-gated API handle.
#[derive(Debug, Clone)]
pub struct MercyGatedApi {
    pub bound_addr: Option<SocketAddr>,
    pub mercy_level: f64,
}

impl MercyGatedApi {
    pub fn new() -> Self {
        Self {
            bound_addr: None,
            mercy_level: 1.0,
        }
    }

    pub fn with_mercy_level(mut self, level: f64) -> Self {
        self.mercy_level = level.clamp(0.0, 1.0);
        self
    }
}

impl Default for MercyGatedApi {
    fn default() -> Self {
        Self::new()
    }
}

/// Start a mercy-gated API server (placeholder — full Axum implementation
/// can be enabled behind the `web-demo` feature in future iterations).
///
/// Currently returns a ready handle without binding a real socket so that
/// the crate remains lightweight and dependency-free by default.
pub fn start_mercy_api_server(addr: Option<SocketAddr>) -> MercyGatedApi {
    println!("[MercyGatedApi] start_mercy_api_server called — mercy level 1.0 (stub ready)");
    MercyGatedApi {
        bound_addr: addr,
        mercy_level: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_defaults() {
        let api = MercyGatedApi::new();
        assert_eq!(api.mercy_level, 1.0);
        assert!(api.bound_addr.is_none());
    }
}
