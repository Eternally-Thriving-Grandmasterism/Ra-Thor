//! adapter.rs
//!
//! Tamper-resistant persistence with HKDF + environment variable support.
//!
//! ## Environment Variable
//!
//! `RA_THOR_HMAC_BASE_KEY`
//!     Optional. If set, used as base material for HKDF key derivation.
//!     Changing this value invalidates previously signed persisted state.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::env;
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 2;

const DEFAULT_BASE_KEY: &[u8] = b"ra-thor-sovereign-self-evolution";
const HKDF_INFO: &[u8] = b"shard-composer-hmac-v1";

const ENV_HMAC_KEY: &str = "RA_THOR_HMAC_BASE_KEY";

// ... rest of the file remains the same
