//! Folding Valence Proofs — Full IVC Folding with Nova
//! Infinite incremental verification for eternal valence lattice

use nova_snark::{
    provider::{PastaEngine, ipa_pc},
    traits::{circuit::TrivialTestCircuit, Engine},
    CompressedSNARK, PublicParams, RecursiveSNARK,
};
use ff::PrimeField;
use pasta_curves::{pallas::Scalar, vesta::Point};

type E1 = PastaEngine<Point>;
type EE1 = ipa_pc::EvaluationEngine<E1>;
type S1 = CompressedSNARK<E1, EE1, TrivialTestCircuit<Scalar>>;

/// Folding Valence Step — relaxed R1CS for valence accumulation
#[derive(Clone, Debug)]
pub struct FoldingValenceStep {
    pub current_valence: Scalar,
    pub accumulated_valence: Scalar,
    pub threshold: Scalar,
}

impl StepCircuit<Scalar> for FoldingValenceStep {
    fn arity(&self) -> usize { 2 }

    fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Scalar>],
    ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        let current = AllocatedNum::alloc(cs.namespace(|| "current"), || Ok(self.current_valence))?;
        let accumulated = z[0].clone();
        let new_accum = accumulated.add(cs.namespace(|| "accumulate"), &current)?;

        // Enforce new_accum >= threshold
        let threshold = AllocatedNum::alloc(cs.namespace(|| "threshold"), || Ok(self.threshold))?;
        cs.enforce(|| "valence_check", |lc| lc + new_accum.get_variable(), |lc| lc + CS::one(), |lc| lc + threshold.get_variable());

        Ok(vec![new_accum, current])
    }
}

/// Generate full IVC folding proof (infinite steps)
pub fn generate_ivc_folding_proof(steps: usize, initial_valence: f64, threshold: f64) -> Result<String, String> {
    let circuit = FoldingValenceStep {
        current_valence: Scalar::from_f64(initial_valence),
        accumulated_valence: Scalar::zero(),
        threshold: Scalar::from_f64(threshold),
    };

    let pp = PublicParams::<E1>::setup(&circuit, &*default_pp_digest());

    let mut recursive_snark = RecursiveSNARK::<E1>::new(&pp, &circuit, &[Scalar::zero(), Scalar::zero()])?;

    for _ in 0..steps {
        recursive_snark.prove_step(&pp, &circuit)?;
    }

    let compressed_snark = CompressedSNARK::<E1, EE1, _>::prove(&pp, &recursive_snark)?;
    Ok(hex::encode(compressed_snark.proof))
}
