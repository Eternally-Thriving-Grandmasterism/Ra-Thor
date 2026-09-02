//! Ra-Thor™ BLS12-381 Signature Aggregation
//! Part of the hybrid classical + post-quantum architecture
//! See docs/HYBRID_CRYPTO_ARCHITECTURE.md for the full strategy

use bls_signatures::{PublicKey, Signature, aggregate, verify};
use crate::patsagi_deliberation::DeliberationSession;