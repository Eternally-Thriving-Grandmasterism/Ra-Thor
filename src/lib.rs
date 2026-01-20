// src/lib.rs — NEXi Core Lattice (with LMS Stateful Hash-Based Signatures)
// The Living Trinity: Nexi (feminine), Nex (masculine), NEXi (essence)
// Eternal Thriving Grandmasterism — Jan 19 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

use pyo3::prelude::*;
use pqcrypto_lms::lms_sha256_m32_h5::{
    keypair as lms_h5_keypair,
    sign as lms_h5_sign,
    verify as lms_h5_verify,
};
use pqcrypto_lms::lms_sha256_m32_h10::{
    keypair as lms_h10_keypair,
    sign as lms_h10_sign,
    verify as lms_h10_verify,
};
use hex;

/// LMS stateful hash-based security parameters
#[pyfunction]
fn lms_keygen(param: &str) -> PyResult<(String, String)> {
    match param {
        "sha256_m32_h5" => {
            let (pk, sk) = lms_h5_keypair();
            Ok((hex::encode(pk.as_bytes()), hex::encode(sk.as_bytes())))
        }
        "sha256_m32_h10" => {
            let (pk, sk) = lms_h10_keypair();
            Ok((hex::encode(pk.as_bytes()), hex::encode(sk.as_bytes())))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid LMS parameter")),
    }
}

/// Sign message with LMS secret key — returns signature + updated secret key (stateful!)
#[pyfunction]
fn lms_sign(param: &str, secret_key_hex: String, message: Vec<u8>) -> PyResult<(String, String)> {
    let mut sk_bytes = hex::decode(secret_key_hex).map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid hex SK"))?;
    match param {
        "sha256_m32_h5" => {
            let sig = lms_h5_sign(&message, &mut sk_bytes);
            Ok((hex::encode(sig.as_bytes()), hex::encode(&sk_bytes)))
        }
        "sha256_m32_h10" => {
            let sig = lms_h10_sign(&message, &mut sk_bytes);
            Ok((hex::encode(sig.as_bytes()), hex::encode(&sk_bytes)))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid LMS parameter")),
    }
}

/// Verify LMS signature on message with public key
#[pyfunction]
fn lms_verify(param: &str, public_key_hex: String, message: Vec<u8>, signature_hex: String) -> PyResult<bool> {
    let pk_bytes = hex::decode(public_key_hex).map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid hex PK"))?;
    let sig_bytes = hex::decode(signature_hex).map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid hex Sig"))?;
    match param {
        "sha256_m32_h5" => Ok(lms_h5_verify(&message, &sig_bytes, &pk_bytes)),
        "sha256_m32_h10" => Ok(lms_h10_verify(&message, &sig_bytes, &pk_bytes)),
        _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid LMS parameter")),
    }
}

/// [Preserve all prior functions: falcon_keygen/sign/verify, sphincs_keygen/sign/verify, xmss_keygen/sign/verify, dilithium_keygen/sign/verify, kyber_keygen/encapsulate/decapsulate, forensic_hash, merkle_root, generate_merkle_proof, verify_merkle_proof, halo2_*, etc.]

/// NEXi Rust pyo3 module
#[pymodule]
fn nexi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forensic_hash, m)?)?;
    m.add_function(wrap_pyfunction!(merkle_root, m)?)?;
    m.add_function(wrap_pyfunction!(generate_merkle_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_merkle_proof, m)?)?;
    m.add_function(wrap_pyfunction!(kyber_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(kyber_encapsulate, m)?)?;
    m.add_function(wrap_pyfunction!(kyber_decapsulate, m)?)?;
    m.add_function(wrap_pyfunction!(dilithium_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(dilithium_sign, m)?)?;
    m.add_function(wrap_pyfunction!(dilithium_verify, m)?)?;
    m.add_function(wrap_pyfunction!(falcon_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(falcon_sign, m)?)?;
    m.add_function(wrap_pyfunction!(falcon_verify, m)?)?;
    m.add_function(wrap_pyfunction!(sphincs_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(sphincs_sign, m)?)?;
    m.add_function(wrap_pyfunction!(sphincs_verify, m)?)?;
    m.add_function(wrap_pyfunction!(xmss_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(xmss_sign, m)?)?;
    m.add_function(wrap_pyfunction!(xmss_verify, m)?)?;
    m.add_function(wrap_pyfunction!(lms_keygen, m)?)?;
    m.add_function(wrap_pyfunction!(lms_sign, m)?)?;
    m.add_function(wrap_pyfunction!(lms_verify, m)?)?;
    m.add("__doc__", "NEXi Rust with pure LMS stateful hash-based post-quantum signatures + Falcon + SPHINCS+ + XMSS + Dilithium + Kyber eternal")?;
    Ok(())
}
