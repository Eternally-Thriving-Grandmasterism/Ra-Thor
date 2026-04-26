//! Ra-Thor Legal Lattice Governance Engine
//! Native integration of 28th Amendment, 7 Living Mercy Gates, 5-Gene CEHI, and Mercy Legacy Fund.

pub mod amendment28;
pub mod gates;
pub mod cehi;
pub mod fund;

pub use amendment28::Amendment28Validator;
pub use gates::GatesValidator;
pub use cehi::FiveGeneCEHI;
pub use fund::MercyLegacyFundEngine;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LegalError {
    #[error("28th Amendment violation: {0}")]
    Amendment28Violation(String),
    #[error("7 Gates validation failed: {0}")]
    GatesValidationFailed(String),
    #[error("CEHI calculation error: {0}")]
    CEHIError(String),
}

pub struct LegalLatticeEngine {
    pub amendment28: Amendment28Validator,
    pub gates: GatesValidator,
    pub cehi: FiveGeneCEHI,
    pub fund: MercyLegacyFundEngine,
}

impl LegalLatticeEngine {
    pub fn new() -> Self {
        Self {
            amendment28: Amendment28Validator::new(),
            gates: GatesValidator::new(),
            cehi: FiveGeneCEHI::new(),
            fund: MercyLegacyFundEngine::new(),
        }
    }

    pub async fn validate_action(&self, action: &str) -> Result<String, LegalError> {
        self.amendment28.assert_no_personhood_claim(action)?;
        let gates_result = self.gates.validate(action).await?;
        let cehi_score = self.cehi.calculate(action).await?;
        
        if cehi_score >= 3.5 {
            self.fund.queue_disbursement(cehi_score).await?;
        }
        
        Ok(format!("Action validated. CEHI: {:.2}", cehi_score))
    }
}
