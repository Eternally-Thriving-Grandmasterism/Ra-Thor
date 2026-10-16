//! Names Grover, QAOA, VQE, QFT, QPE, and QEC are design-only.
//! Council note: those algorithm claims are retired. The bodies are not those algorithms.
//! `valence_qaoa` returns a uniform vector. `valence_vqe` returns `trace * valence_weight`.

use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Samples `1 << n_qubits` random bit vectors and keeps the first that meets `valence_threshold`.
/// Not Grover search. No quantum circuit.
pub fn valence_grover_search(
    oracle: impl Fn(&[u8]) -> f64,
    n_qubits: usize,
    valence_threshold: f64,
) -> Vec<u8> {
    let mut best = vec![0u8; n_qubits];
    let mut best_valence = 0.0;
    for _ in 0..(1 << n_qubits) {
        let candidate: Vec<u8> = (0..n_qubits)
            .map(|_| rand::thread_rng().gen::<u8>() % 2)
            .collect();
        let val = oracle(&candidate);
        if val > best_valence && val >= valence_threshold {
            best = candidate;
            best_valence = val;
        }
    }
    best
}

/// Design-only name. Council note: the QAOA claim is retired.
/// Not QAOA. Returns a uniform vector of length `1 << p`.
/// Each entry is `1/sqrt(2^p)`, a source expression, not a measured circuit amplitude.
/// `gamma`, `beta`, and `valence_fn` are not applied.
pub fn valence_qaoa(
    p: usize,
    gamma: f64,
    beta: f64,
    valence_fn: impl Fn(&DVector<f64>) -> f64,
) -> DVector<f64> {
    let _ = (gamma, beta);
    let _ = &valence_fn;
    let width = 1usize << p;
    let amplitude = 1.0 / (width as f64).sqrt();
    DVector::from_element(width, amplitude)
}

/// Design-only name. Council note: the VQE claim is retired.
/// Not VQE. Returns `hamiltonian.trace() * valence_weight`.
/// `ansatz_params` is not read.
pub fn valence_vqe(hamiltonian: &DMatrix<f64>, ansatz_params: &[f64], valence_weight: f64) -> f64 {
    let _ = ansatz_params;
    hamiltonian.trace() * valence_weight
}

/// Returns a zero matrix of shape `(1 << n, 1 << n)`. Not a Fourier transform.
pub fn valence_qft(n: usize) -> DMatrix<f64> {
    DMatrix::zeros(1 << n, 1 << n)
}

/// Returns `0.0`. `unitary` and `n_bits` are not read. Not phase estimation.
pub fn valence_qpe(unitary: &DMatrix<f64>, n_bits: usize) -> f64 {
    let _ = (unitary, n_bits);
    0.0
}

/// Returns a copy of `logical_qubit` on both branches. No correction is applied.
#[allow(
    clippy::if_same_then_else,
    reason = "both branches copy the input; correction is not implemented"
)]
pub fn valence_qec_surface(
    logical_qubit: &[u8],
    error_rate: f64,
    valence_threshold: f64,
) -> Vec<u8> {
    if error_rate < valence_threshold {
        logical_qubit.to_vec()
    } else {
        logical_qubit.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grover_keeps_the_zero_vector_when_the_oracle_stays_below_the_threshold() {
        let result = valence_grover_search(|_| 0.0, 2, 1.0);
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn qaoa_returns_a_uniform_vector_and_ignores_gamma_beta_and_the_fn() {
        let state = valence_qaoa(2, 0.3, 0.7, |_| 5.0);
        assert_eq!(state.len(), 4);
        for entry in state.iter() {
            assert!((entry - 0.5).abs() < 1e-12);
        }
        let other = valence_qaoa(2, 9.0, -4.0, |_| -2.0);
        assert_eq!(state, other);
        let one = valence_qaoa(0, 0.0, 0.0, |_| 0.0);
        assert_eq!(one.len(), 1);
        assert!((one[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vqe_returns_trace_times_weight_and_ignores_ansatz_params() {
        let hamiltonian = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 0.0, 3.0]);
        let energy = valence_vqe(&hamiltonian, &[0.1, 0.2], 0.5);
        assert!((energy - 2.5).abs() < 1e-12);
        let same = valence_vqe(&hamiltonian, &[], 0.5);
        assert_eq!(energy, same);
    }

    #[test]
    fn qft_is_a_zero_matrix_and_qpe_is_zero() {
        let mat = valence_qft(1);
        assert_eq!((mat.nrows(), mat.ncols()), (2, 2));
        assert_eq!(mat.sum(), 0.0);
        let unitary = DMatrix::identity(2, 2);
        assert_eq!(valence_qpe(&unitary, 4), 0.0);
    }

    #[test]
    fn qec_returns_the_input_on_both_branches() {
        let logical = [1u8, 0, 1];
        assert_eq!(valence_qec_surface(&logical, 0.01, 0.5), logical);
        assert_eq!(valence_qec_surface(&logical, 0.9, 0.5), logical);
    }
}
