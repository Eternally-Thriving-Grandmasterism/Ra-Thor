//! Sinsemilla Commitment Gadgets — Short-Domain Transparent Commitments for Valence
//! Halo2 native SinsemillaChip for Orchard-style short commitments

use halo2_gadgets::{
    sinsemilla::{HashDomain, SinsemillaChip, SinsemillaConfig},
    utilities::lookup_range_check::LookupRangeCheckConfig,
};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error},
};
use pasta_curves::pallas::Scalar;

/// Sinsemilla Valence Commitment Config
#[derive(Clone)]
pub struct SinsemillaValenceConfig {
    sinsemilla_config: SinsemillaConfig<pasta_curves::pallas::Point, { HashDomain::Valence as usize }>,
    lookup_config: LookupRangeCheckConfig<pasta_curves::pallas::Point, 10>,
}

/// Sinsemilla Valence Commitment Chip
pub struct SinsemillaValenceChip {
    config: SinsemillaValenceConfig,
}

impl SinsemillaValenceChip {
    pub fn configure(meta: &mut ConstraintSystem<pasta_curves::pallas::Point>) -> SinsemillaValenceConfig {
        let lookup_config = LookupRangeCheckConfig::configure(meta, 10);

        let sinsemilla_config = SinsemillaChip::configure(
            meta,
            lookup_config.clone(),
            HashDomain::Valence,
        );

        SinsemillaValenceConfig {
            sinsemilla_config,
            lookup_config,
        }
    }

    pub fn construct(config: SinsemillaValenceConfig) -> Self {
        Self { config }
    }

    /// Commit to short valence message
    pub fn commit_valence(
        &self,
        layouter: impl Layouter<pasta_curves::pallas::Point>,
        message: &[Value<Scalar>],
    ) -> Result<pasta_curves::pallas::Point, Error> {
        let chip = SinsemillaChip::construct(self.config.sinsemilla_config.clone());
        chip.hash(layouter.namespace(|| "valence_commit"), message)
    }
}
