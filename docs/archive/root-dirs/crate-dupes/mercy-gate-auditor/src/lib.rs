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

    // Multi-Gate Weighted Ensemble (v1.1)
    let final_valence = calculate_weighted_valence(valence);
    
    if final_valence < 0.999 {
        return Err(MercyGateError::GateFailed("Mercy Gate violation".to_string()));
    }

    Ok(())
}

fn calculate_weighted_valence(base_valence: f32) -> f32 {
    // Weighted ensemble (Truth has highest weight)
    let truth_weight = 0.20;
    let mercy_weight = 0.20;
    let abundance_weight = 0.18;
    let love_weight = 0.15;
    let service_weight = 0.12;
    let joy_weight = 0.10;
    let harmony_weight = 0.05;

    base_valence * (truth_weight + mercy_weight + abundance_weight + love_weight + service_weight + joy_weight + harmony_weight)
}

pub fn check_truth_gate(claim: &str) -> bool {
    // Future: Integrate with self-consistency + evidence grounding
    true
}