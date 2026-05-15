use thiserror::Error;

#[derive(Debug, Error)]
pub enum MercyGateError {
    #[error("Valence too low: {0}")]
    ValenceTooLow(f32),
    #[error("Gate failed: {0}")]
    GateFailed(String),
}

pub struct MercyGateResult {
    pub passed: bool,
    pub valence: f32,
    pub failed_gate: Option<String>,
}

pub fn audit_output(output: &str, valence: f32) -> Result<(), MercyGateError> {
    if valence < 0.999 {
        return Err(MercyGateError::ValenceTooLow(valence));
    }

    // Placeholder for full 7-gate check
    // In production, this would call each gate
    if output.contains("I don't know") || output.contains("uncertain") {
        // Allow uncertainty
        return Ok(());
    }

    Ok(())
}

pub fn check_truth_gate(claim: &str) -> bool {
    // Future: Integrate with self-consistency + evidence grounding
    true
}