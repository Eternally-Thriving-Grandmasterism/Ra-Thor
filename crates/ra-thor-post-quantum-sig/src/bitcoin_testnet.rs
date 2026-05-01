//! Bitcoin Testnet Integration — RHPQS v0.1.0
//! Mercy-Gated Post-Quantum Signatures for Bitcoin Transactions
//!
//! This module enables signing Bitcoin testnet transactions using RHPQS.

use crate::{RHPQSEngine, RHPQSKey, RHPQSSignature, RHPQSError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinTestnetTransaction {
    pub txid: String,
    pub amount: u64,           // satoshis
    pub recipient: String,     // Bitcoin address
    pub signature: Option<RHPQSSignature>,
}

pub struct BitcoinTestnetSigner<'a> {
    engine: &'a RHPQSEngine,
    key: &'a RHPQSKey,
}

impl<'a> BitcoinTestnetSigner<'a> {
    pub fn new(engine: &'a RHPQSEngine, key: &'a RHPQSKey) -> Self {
        Self { engine, key }
    }

    /// Sign a Bitcoin testnet transaction with RHPQS
    pub async fn sign_transaction(
        &self,
        tx: &mut BitcoinTestnetTransaction,
    ) -> Result<RHPQSSignature, RHPQSError> {
        let message = format!(
            "Bitcoin Testnet TX: {} to {} for {} sats",
            tx.txid, tx.recipient, tx.amount
        );

        let signature = self.engine.sign(self.key, message.as_bytes()).await?;
        tx.signature = Some(signature.clone());

        Ok(signature)
    }

    /// Verify a signed Bitcoin testnet transaction
    pub fn verify_transaction(&self, tx: &BitcoinTestnetTransaction) -> Result<bool, RHPQSError> {
        if let Some(sig) = &tx.signature {
            let message = format!(
                "Bitcoin Testnet TX: {} to {} for {} sats",
                tx.txid, tx.recipient, tx.amount
            );
            self.engine.verify(sig, message.as_bytes())
        } else {
            Ok(false)
        }
    }
}
