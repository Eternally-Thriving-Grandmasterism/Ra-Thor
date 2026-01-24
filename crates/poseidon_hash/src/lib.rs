//! PoseidonHash — zk-Friendly Hashing Module
//! Ultramasterful integration for Merkle trees, commitments, SoulScan waveform

use halo2_gadgets::poseidon::{
    Pow5Config as PoseidonConfig, 
    PoseidonChip, 
    PoseidonSponge,
};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
};
use pasta_curves::pallas::Scalar;

pub struct PoseidonHash {
    config: PoseidonConfig<Scalar, 3, 2>,
}

impl PoseidonHash {
    pub fn new() -> Self {
        // t=3, rate=2 Poseidon (standard zk-friendly)
        let config = PoseidonConfig::default();
        PoseidonHash { config }
    }

    /// Hash arbitrary input for Merkle/commitments
    pub fn hash(&self, layouter: impl Layouter<Scalar>, inputs: &[Value<Scalar>]) -> Result<Scalar, Error> {
        let chip = PoseidonChip::<Scalar, 3, 2>::construct(self.config.clone());
        chip.hash(layouter.namespace(|| "poseidon_hash"), inputs)
    }

    /// Sponge mode for SoulScan waveform hashing
    pub fn sponge_hash(&self, layouter: impl Layouter<Scalar>, inputs: &[Value<Scalar>]) -> Result<Scalar, Error> {
        let mut sponge = PoseidonSponge::new(self.config.clone());
        sponge.absorb(layouter.namespace(|| "sponge_absorb"), inputs)?;
        sponge.squeeze(layouter.namespace(|| "sponge_squeeze"))
    }
}
