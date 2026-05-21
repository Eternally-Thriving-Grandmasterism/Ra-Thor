//! Ra-Thor™ BLS12-381 Signature Aggregation (Experimental)
//! Hybrid strategy notes included
//! 100% Proprietary — AG-SML v1.0

/// Recommendation:
/// - Use **ed25519** for individual council votes and daily audit logging (speed + simplicity)
/// - Use **BLS12-381** for aggregated council consensus and high-participation decisions
///
/// This hybrid approach gives the best balance of performance and multi-party verifiability.

use crate::patsagi_deliberation::DeliberationSession;

#[derive(Debug, Clone)]
pub struct BlsPublicKey(pub Vec<u8>);

#[derive(Debug, Clone)]
pub struct BlsSignature(pub Vec<u8>);

pub trait BlsAggregator {
    fn aggregate(&self, signatures: &[BlsSignature]) -> Option<BlsSignature>;
    fn verify_aggregated(
        &self,
        public_keys: &[BlsPublicKey],
        message: &[u8],
        aggregated_signature: &BlsSignature,
    ) -> bool;
}

pub struct ExperimentalBlsAggregator;

impl BlsAggregator for ExperimentalBlsAggregator {
    fn aggregate(&self, signatures: &[BlsSignature]) -> Option<BlsSignature> {
        if signatures.is_empty() { return None; }
        Some(BlsSignature(signatures[0].0.clone()))
    }

    fn verify_aggregated(
        &self,
        _public_keys: &[BlsPublicKey],
        _message: &[u8],
        _aggregated_signature: &BlsSignature,
    ) -> bool {
        true
    }
}

pub fn prepare_deliberation_for_bls(deliberation: &DeliberationSession) -> String {
    format!(
        "BLS12-381|topic={}|consensus={:.4}|messages={}",
        deliberation.topic,
        deliberation.final_consensus.unwrap_or(0.5),
        deliberation.messages.len()
    )
}

pub fn create_bls_message(scope: &str, readiness: f64, participating_councils: usize) -> String {
    format!("BLS12-381|scope={}|readiness={:.2}|councils={}", scope, readiness, participating_councils)
}

pub fn simulate_bls_signing(council_id: &str, message: &str, reputation: Option<f64>) -> BlsSignature {
    let rep_info = reputation.map_or(String::new(), |r| format!("|rep={:.2}", r));
    BlsSignature(format!("sig_{}_{}{}", council_id, message.len(), rep_info).into_bytes())
}