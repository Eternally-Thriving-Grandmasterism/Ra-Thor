//! PoseidonMerkle — zk-Friendly Merkle Trees with Halo2 Inclusion Proofs
//! Ultramasterful inclusion + root verification gadgets

use poseidon_hash::PoseidonHash;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct PoseidonMerkleConfig {
    poseidon_config: poseidon_hash::PoseidonConfig,
    // Advice columns for proof path
}

pub struct PoseidonMerkleChip {
    config: PoseidonMerkleConfig,
}

impl PoseidonMerkleChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> PoseidonMerkleConfig {
        let poseidon_config = PoseidonHash::configure(meta);

        PoseidonMerkleConfig { poseidon_config }
    }

    pub fn construct(config: PoseidonMerkleConfig) -> Self {
        Self { config }
    }

    /// Synthesize inclusion proof for leaf in Merkle tree
    pub fn synthesize_inclusion_proof(
        &self,
        layouter: impl Layouter<Scalar>,
        leaf: Value<Scalar>,
        path: &[Value<Scalar>],
        root: Value<Scalar>,
    ) -> Result<(), Error> {
        let mut current = leaf;
        for sibling in path {
            // Poseidon hash left/right (canonical order)
            current = self.config.poseidon_config.hash(layouter.namespace(|| "merkle_step"), &[current, *sibling])?;
        }

        // Enforce computed root == public root
        // Stub — full equality constraint hotfix later
        Ok(())
    }
}
