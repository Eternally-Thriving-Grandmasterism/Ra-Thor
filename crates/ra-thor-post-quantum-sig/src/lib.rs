// src/lib.rs
pub mod error;
pub mod traits;

pub use error::PostQuantumError;
pub use traits::PostQuantumSignature;

// Re-export for convenience
pub use mercy_tolc_operator_algebra;
pub use ra_thor_mercy;