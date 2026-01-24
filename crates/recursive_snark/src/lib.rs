//! RecursiveSNARK — Infinite Proof-of-Proof Composition
//! Ultramasterful Halo2 folding gadgets for eternal recursion

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2_gadgets::recursive::RecursiveCompositionChip;
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct RecursiveSNARKConfig {
    composition_config: RecursiveCompositionConfig,
}

pub struct RecursiveSNARKChip {
    config: RecursiveSNARKConfig,
}

impl RecursiveSNARKChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> RecursiveSNARKConfig {
        let composition_config = RecursiveCompositionChip::configure(meta);

        RecursiveSNARKConfig { composition_config }
    }

    pub fn construct(config: RecursiveSNARKConfig) -> Self {
        Self { config }
    }

    /// Compose prior proof into current — infinite recursion
    pub fn compose_recursive_proof(
        &self,
        layouter: impl Layouter<Scalar>,
        prior_proof: Value<Scalar>,
        current_proof: Value<Scalar>,
    ) -> Result<Scalar, Error> {
        let composition = RecursiveCompositionChip::construct(self.config.composition_config.clone());
        composition.compose(layouter.namespace(|| "recursive_snark"), prior_proof, current_proof)
    }
}
