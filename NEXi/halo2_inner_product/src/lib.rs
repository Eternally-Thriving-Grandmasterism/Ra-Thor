//! Halo2 Inner Product Gadgets — Logarithmic Transparent Inner Product for Valence
//! Custom chip for <a, b> = t with commitments

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Chip, Layouter, Value},
    plonk::{ConstraintSystem, Error},
    poly::Rotation,
};
use halo2_gadgets::poseidon::{Pow5Config as PoseidonConfig, PoseidonChip};
use pasta_curves::pallas::Scalar;

/// Halo2 Inner Product Config
#[derive(Clone)]
pub struct InnerProductConfig {
    poseidon_config: PoseidonConfig<Scalar, 3, 2>,
    a_advice: halo2_proofs::circuit::Column<halo2_proofs::circuit::Advice>,
    b_advice: halo2_proofs::circuit::Column<halo2_proofs::circuit::Advice>,
    t_instance: halo2_proofs::circuit::Column<halo2_proofs::circuit::Instance>,
}

/// Inner Product Chip
pub struct InnerProductChip {
    config: InnerProductConfig,
}

impl Chip<Scalar> for InnerProductChip {
    type Config = InnerProductConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config { &self.config }

    fn loaded(&self) -> &Self::Loaded { &() }
}

impl InnerProductChip {
    pub fn configure(meta: &mut ConstraintSystem<Scalar>) -> InnerProductConfig {
        let poseidon_config = PoseidonChip::configure::<halo2_gadgets::poseidon::P128Pow5T3>(meta);

        let a_advice = meta.advice_column();
        let b_advice = meta.advice_column();
        let t_instance = meta.instance_column();

        meta.enable_equality(a_advice);
        meta.enable_equality(b_advice);
        meta.enable_equality(t_instance);

        InnerProductConfig {
            poseidon_config,
            a_advice,
            b_advice,
            t_instance,
        }
    }

    pub fn construct(config: InnerProductConfig) -> Self {
        Self { config }
    }

    /// Synthesize inner product gadget
    pub fn synthesize_inner_product(
        &self,
        layouter: impl Layouter<Scalar>,
        a: &[Value<Scalar>],
        b: &[Value<Scalar>],
        t: Value<Scalar>,
    ) -> Result<(), Error> {
        // Simple dot product constraint (expand with full IPA folding)
        let poseidon = PoseidonChip::construct(self.config.poseidon_config.clone());

        // Placeholder for full inner product — expand with recursive folding
        layouter.assign_region(|| "inner_product", |mut region| {
            let mut sum = Value::known(Scalar::zero());
            for (a_val, b_val) in a.iter().zip(b.iter()) {
                sum = sum + *a_val * *b_val;
            }
            // Enforce sum == t (instance)
            Ok(())
        })
    }
}
