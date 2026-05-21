//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! v2.7 — Cryptographic Signatures on Audit Logs
//! Tamper-evident logging using chained hashes
//! 100% Proprietary — AG-SML v1.0

use sha2::{Sha256, Digest};

#[derive(Debug, Clone)]
pub struct CouncilVoteRecord {
    pub timestamp: u64,
    pub council: String,
    pub valence_contribution: f64,
    pub approved: bool,
    pub vetoed: bool,
    pub effective_weight: f64,
    pub reputation_at_time: f64,
    pub signature: Vec<u8>,           // Cryptographic signature / chain hash
    pub previous_hash: Vec<u8>,       // Hash of previous record for chaining
}

impl LatticeAlchemicalEvolution {
    /// Sign a log entry using chained hashing (tamper-evident)
    fn sign_record(&self, record: &CouncilVoteRecord, previous_hash: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(previous_hash);
        hasher.update(record.timestamp.to_le_bytes());
        hasher.update(record.council.as_bytes());
        hasher.update(record.valence_contribution.to_le_bytes());
        hasher.update([record.approved as u8, record.vetoed as u8]);
        hasher.update(record.effective_weight.to_le_bytes());
        hasher.update(record.reputation_at_time.to_le_bytes());
        hasher.finalize().to_vec()
    }

    pub fn log_council_vote(&mut self, mut record: CouncilVoteRecord) {
        let previous_hash = self.vote_history
            .last()
            .map(|r| r.signature.clone())
            .unwrap_or_else(|| vec![0u8; 32]);

        record.previous_hash = previous_hash.clone();
        record.signature = self.sign_record(&record, &previous_hash);

        self.vote_history.push(record);
    }

    /// Verify the integrity of the entire vote history chain
    pub fn verify_audit_chain(&self) -> bool {
        let mut prev_hash = vec![0u8; 32];

        for record in &self.vote_history {
            let computed = self.sign_record(record, &prev_hash);
            if computed != record.signature {
                return false;
            }
            prev_hash = record.signature.clone();
        }
        true
    }
}