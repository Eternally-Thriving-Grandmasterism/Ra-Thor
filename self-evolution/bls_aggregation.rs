//! Ra-Thor™ BLS12-381 Signature Aggregation
//! Using bls-signatures crate for real BLS12-381 operations
//! 100% Proprietary — AG-SML v1.0

use bls_signatures::{PrivateKey, PublicKey, Signature, aggregate, verify};
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
        let sigs: Vec<Signature> = signatures
            .iter()
            .filter_map(|s| Signature::from_bytes(&s.0).ok())
            .collect();

        aggregate(&sigs)
            .map(|agg| BlsSignature(agg.as_bytes().to_vec()))
            .ok()
    }

    fn verify_aggregated(
        &self,
        public_keys: &[BlsPublicKey],
        message: &[u8],
        aggregated_signature: &BlsSignature,
    ) -> bool {
        let pks: Vec<PublicKey> = public_keys
            .iter()
            .filter_map(|pk| PublicKey::from_bytes(&pk.0).ok())
            .collect();

        if let Ok(sig) = Signature::from_bytes(&aggregated_signature.0) {
            return verify(&sig, message, &pks);
        }
        false
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