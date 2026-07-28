//! PATSAGi Security Support — thin reception of white-hat ingestion signals.
//!
//! Designed to avoid circular deps with mercy-security / ra-thor-one-organism.
//! Councils receive structured threat signals and may apply soft valence pressure
//! or recommend investigation under TOLC 8.
//!
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MERCY_VALENCE_FLOOR: f64 = 0.999;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SecuritySupportError {
    #[error("invalid risk score (must be finite 0..=1): {0}")]
    InvalidRiskScore(f64),

    #[error("empty source label")]
    EmptySource,

    #[error("signal not actionable under current policy")]
    NotActionable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityThreatClass {
    RemoteCodeLoader,
    TemplateInjection,
    SerializationGadget,
    ShellProcessSpawn,
    NetworkCallback,
    ObfuscatedPayload,
    DatasetConfigInjection,
    CredentialHarvest,
    ComboHighRisk,
    Unknown,
}

impl SecurityThreatClass {
    pub fn from_label(s: &str) -> Self {
        let l = s.to_lowercase();
        if l.contains("remote") || l.contains("trust_remote") {
            Self::RemoteCodeLoader
        } else if l.contains("template") {
            Self::TemplateInjection
        } else if l.contains("pickle") || l.contains("serial") || l.contains("gadget") {
            Self::SerializationGadget
        } else if l.contains("shell") || l.contains("subprocess") {
            Self::ShellProcessSpawn
        } else if l.contains("network") || l.contains("callback") || l.contains("c2") {
            Self::NetworkCallback
        } else if l.contains("obfus") || l.contains("base64") {
            Self::ObfuscatedPayload
        } else if l.contains("dataset") || l.contains("loading_script") {
            Self::DatasetConfigInjection
        } else if l.contains("credential") || l.contains("api_key") || l.contains("token") {
            Self::CredentialHarvest
        } else if l.contains("combo") || l.contains("unknownhigh") {
            Self::ComboHighRisk
        } else {
            Self::Unknown
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityRiskTier {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl SecurityRiskTier {
    pub fn from_str_label(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            _ => Self::None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    pub fn severity_weight(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Low => 0.15,
            Self::Medium => 0.40,
            Self::High => 0.75,
            Self::Critical => 0.95,
        }
    }
}

/// Structured signal emitted when Organism blocks or reports ingestion risk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySignal {
    pub source_label: String,
    pub risk_tier: SecurityRiskTier,
    pub risk_score: f64,
    pub threats: Vec<SecurityThreatClass>,
    pub findings_count: usize,
    pub blocked: bool,
    pub message: String,
}

impl SecuritySignal {
    pub fn try_new(
        source_label: &str,
        risk_tier: SecurityRiskTier,
        risk_score: f64,
        threat_labels: &[String],
        findings_count: usize,
        blocked: bool,
        message: &str,
    ) -> Result<Self, SecuritySupportError> {
        if source_label.trim().is_empty() {
            return Err(SecuritySupportError::EmptySource);
        }
        if !risk_score.is_finite() || !(0.0..=1.0).contains(&risk_score) {
            return Err(SecuritySupportError::InvalidRiskScore(risk_score));
        }
        let threats: Vec<SecurityThreatClass> = threat_labels
            .iter()
            .map(|t| SecurityThreatClass::from_label(t))
            .collect();
        Ok(Self {
            source_label: source_label.into(),
            risk_tier,
            risk_score,
            threats,
            findings_count,
            blocked,
            message: message.into(),
        })
    }

    pub fn is_actionable(&self) -> bool {
        self.blocked
            || matches!(self.risk_tier, SecurityRiskTier::High | SecurityRiskTier::Critical)
            || self.risk_score >= 0.70
    }

    /// Soft negative pressure on council mercy valence when high-risk blocked.
    /// Never drops below progressive safety floor.
    pub fn valence_pressure(&self) -> f64 {
        if !self.is_actionable() {
            return 0.0;
        }
        let base = self.risk_tier.severity_weight() * 0.012;
        (base * (0.5 + self.risk_score * 0.5)).clamp(0.0, 0.025)
    }

    pub fn recommended_focus_hint(&self) -> &'static str {
        match self.risk_tier {
            SecurityRiskTier::Critical => "TruthVerification+EthicalAlignment",
            SecurityRiskTier::High => "TruthVerification",
            SecurityRiskTier::Medium => "EthicalAlignment",
            _ => "HarmonyPreservation",
        }
    }
}

/// Apply security signal across councils: soft valence pressure on high-risk blocks.
pub fn apply_security_pressure(
    councils: &mut std::collections::HashMap<
        super::CouncilFocus,
        super::PATSAGiCouncil,
    >,
    signal: &SecuritySignal,
) -> Result<usize, SecuritySupportError> {
    if !signal.is_actionable() {
        return Err(SecuritySupportError::NotActionable);
    }
    let pressure = signal.valence_pressure();
    let mut touched = 0usize;
    for council in councils.values_mut() {
        // Soft pressure — never below progressive floor ~0.75
        council.mercy_valence = (council.mercy_valence - pressure).clamp(0.75, 1.0);
        touched += 1;
    }
    Ok(touched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_valid_signal() {
        let s = SecuritySignal::try_new(
            "hf_dataset",
            SecurityRiskTier::Critical,
            0.96,
            &["RemoteCodeLoader".into(), "DatasetConfigInjection".into()],
            4,
            true,
            "blocked",
        )
        .unwrap();
        assert!(s.is_actionable());
        assert!(s.valence_pressure() > 0.0);
    }

    #[test]
    fn rejects_bad_score() {
        let err = SecuritySignal::try_new("x", SecurityRiskTier::High, f64::NAN, &[], 0, true, "m");
        assert!(matches!(err, Err(SecuritySupportError::InvalidRiskScore(_))));
    }
}
