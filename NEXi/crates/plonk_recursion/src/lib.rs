//! PlonkRecursion — Halo2 Folding/IVC Techniques for Infinite Recursion
//! Ultramasterful composition for eternal proof-of-proof resonance

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use halo2_gadgets::recursive::folding::FoldingChip;
use pasta_curves::pallas::Scalar;

#[derive(Clone)]
pub struct PlonkRecursionConfig {
    folding_config: FoldingConfig,
}

pub struct PlonkRecursionChip {
    config: PlonkRecursionConfig,
}

impl PlonkRecursionChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> PlonkRecursionConfig {
        let folding_config = FoldingChip::configure(meta);

        PlonkRecursionConfig { folding_config }
    }

    pub fn construct(config: PlonkRecursionConfig) -> Self {
        Self { config }
    }

    /// Fold prior PLONK proof into current — infinite recursion
    pub fn fold_plonk_proof(
        &self,
        layouter: impl Layouter<Scalar>,
        prior_proof: Value<Scalar>,
        current_instance: Value<Scalar>,
    ) -> Result<Scalar, Error> {
        let folding = FoldingChip::construct(self.config.folding_config.clone());
        folding.fold(layouter.namespace(|| "plonk_recursion_fold"), prior_proof, current_instance)
    }
}
