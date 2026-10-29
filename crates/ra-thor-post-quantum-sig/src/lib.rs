//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! placeholder, not working crypto
pub mod error;
pub mod traits;
pub mod hybrid;
pub mod algorithms;

pub use error::PostQuantumError;
pub use traits::PostQuantumSignature;
pub use hybrid::HybridSigner;
pub use algorithms::dilithium::DilithiumSigner;

// Re-exports for convenience
pub use mercy_tolc_operator_algebra;
pub use ra_thor_mercy;