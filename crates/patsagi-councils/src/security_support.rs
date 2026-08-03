//! PATSAGi Security Support — thin reception of white-hat ingestion signals.
//!
//! Designed to avoid circular deps with mercy-security / ra-thor-one-organism.
//! Councils receive structured threat signals, apply soft valence pressure under
//! domain-aware focus weights, and emit formal deliberation verdicts under TOLC 8.
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

/// Host domain envelope (mirrors mercy-security presets without depending on that crate).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SecurityDomainProfile {
    #[default]
    StrictWhitehat,
    Research,
    Enterprise,
    Education,
    CreativeContent,
}

impl SecurityDomainProfile {
    pub fn from_label(s: &str) -> Self {
        let l = s.to_lowercase();
        if l.contains("research") || l.contains("hub") {
            Self::Research
        } else if l.contains("enterprise") || l.contains("copilot") {
            Self::Enterprise
        } else if l.contains("edu") || l.contains("classroom") || l.contains("lab") {
            Self::Education
        } else if l.contains("creative") || l.contains("content") {
            Self::CreativeContent
        } else {
            Self::StrictWhitehat
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StrictWhitehat => "strict_whitehat",
            Self::Research => "domain_research",
            Self::Enterprise => "domain_enterprise",
            Self::Education => "domain_education",
            Self::CreativeContent => "domain_creative_content",
        }
    }

    /// Multiplier on valence pressure (education/research more sensitive).
    pub fn pressure_multiplier(&self) -> f64 {
        match self {
            Self::Education => 1.15,
            Self::Research => 1.10,
            Self::Enterprise => 1.05,
            Self::CreativeContent => 1.0,
            Self::StrictWhitehat => 1.0,
        }
    }
}

/// Formal council verdict on a white-hat security event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityCouncilVerdict {
    /// Unattended admit remains forbidden; human review optional.
    UpholdBlock,
    /// Block stands; Investigator/Debugger path recommended.
    UpholdBlockInvestigate,
    /// Only for non-actionable / low signals — no policy change.
    NoAction,
    /// Signal malformed or not actionable under policy.
    RejectSignal,
}

impl SecurityCouncilVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UpholdBlock => "uphold_block",
            Self::UpholdBlockInvestigate => "uphold_block_investigate",
            Self::NoAction => "no_action",
            Self::RejectSignal => "reject_signal",
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
    pub domain: SecurityDomainProfile,
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
        Self::try_new_with_domain(
            source_label,
            risk_tier,
            risk_score,
            threat_labels,
            findings_count,
            blocked,
            message,
            SecurityDomainProfile::StrictWhitehat,
        )
    }

    pub fn try_new_with_domain(
        source_label: &str,
        risk_tier: SecurityRiskTier,
        risk_score: f64,
        threat_labels: &[String],
        findings_count: usize,
        blocked: bool,
        message: &str,
        domain: SecurityDomainProfile,
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
            domain,
        })
    }

    pub fn is_actionable(&self) -> bool {
        self.blocked
            || matches!(self.risk_tier, SecurityRiskTier::High | SecurityRiskTier::Critical)
            || self.risk_score >= 0.70
    }

    /// Soft negative pressure on council mercy valence when high-risk blocked.
    /// Never drops below progressive safety floor in apply_security_pressure.
    pub fn valence_pressure(&self) -> f64 {
        if !self.is_actionable() {
            return 0.0;
        }
        let base = self.risk_tier.severity_weight() * 0.012;
        let raw = (base * (0.5 + self.risk_score * 0.5)).clamp(0.0, 0.025);
        (raw * self.domain.pressure_multiplier()).clamp(0.0, 0.03)
    }

    pub fn recommended_focus_hint(&self) -> &'static str {
        match self.risk_tier {
            SecurityRiskTier::Critical => "TruthVerification+EthicalAlignment",
            SecurityRiskTier::High => "TruthVerification",
            SecurityRiskTier::Medium => "EthicalAlignment",
            _ => "HarmonyPreservation",
        }
    }

    pub fn decide_verdict(&self) -> SecurityCouncilVerdict {
        if !self.is_actionable() {
            return SecurityCouncilVerdict::NoAction;
        }
        match self.risk_tier {
            SecurityRiskTier::Critical => SecurityCouncilVerdict::UpholdBlockInvestigate,
            SecurityRiskTier::High => SecurityCouncilVerdict::UpholdBlockInvestigate,
            SecurityRiskTier::Medium if self.blocked => SecurityCouncilVerdict::UpholdBlock,
            _ if self.blocked => SecurityCouncilVerdict::UpholdBlock,
            _ => SecurityCouncilVerdict::NoAction,
        }
    }

    /// Per-council focus weight: Truth / Ethics / QuantumEthics absorb more pressure.
    pub fn focus_pressure_weight(focus: &super::CouncilFocus) -> f64 {
        use super::CouncilFocus::*;
        match focus {
            TruthVerification => 1.25,
            EthicalAlignment => 1.20,
            QuantumEthics => 1.15,
            EternalCompassion => 1.05,
            HarmonyPreservation => 1.0,
            PostScarcityEnforcement => 1.0,
            _ => 0.85,
        }
    }
}

/// Apply security signal across councils with focus-weighted soft valence pressure.
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
    let base_pressure = signal.valence_pressure();
    let mut touched = 0usize;
    for (focus, council) in councils.iter_mut() {
        let w = SecuritySignal::focus_pressure_weight(focus);
        let pressure = (base_pressure * w).clamp(0.0, 0.035);
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
        assert_eq!(s.decide_verdict(), SecurityCouncilVerdict::UpholdBlockInvestigate);
    }

    #[test]
    fn education_domain_raises_pressure() {
        let base = SecuritySignal::try_new(
            "lab",
            SecurityRiskTier::High,
            0.85,
            &["RemoteCodeLoader".into()],
            2,
            true,
            "blocked",
        )
        .unwrap();
        let edu = SecuritySignal::try_new_with_domain(
            "lab",
            SecurityRiskTier::High,
            0.85,
            &["RemoteCodeLoader".into()],
            2,
            true,
            "blocked",
            SecurityDomainProfile::Education,
        )
        .unwrap();
        assert!(edu.valence_pressure() > base.valence_pressure());
    }

    #[test]
    fn rejects_bad_score() {
        let err = SecuritySignal::try_new("x", SecurityRiskTier::High, f64::NAN, &[], 0, true, "m");
        assert!(matches!(err, Err(SecuritySupportError::InvalidRiskScore(_))));
    }

    #[test]
    fn domain_from_label() {
        assert_eq!(SecurityDomainProfile::from_label("classroom-lab"), SecurityDomainProfile::Education);
        assert_eq!(SecurityDomainProfile::from_label("research-hub"), SecurityDomainProfile::Research);
    }
}
