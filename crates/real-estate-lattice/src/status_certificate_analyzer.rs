//! Status Certificate Analyzer for Ontario Condominiums
//!
//! Now supports optional Redis Streams invalidation publishing
//! when high-risk findings (special assessments, litigation) are detected.

#[cfg(feature = "redis")]
use crate::avm_cache_invalidation::RedisStreamPublisher;

#[derive(Debug, Clone)]
pub struct StatusCertificateAnalysis {
    pub corporation_name: String,
    pub reserve_fund_balance: f64,
    pub special_assessments_pending: bool,
    pub litigation_risk: bool,
    pub insurance_status: String,
    pub rules_violations: Vec<String>,
    pub overall_risk_level: String,
    pub warnings: Vec<String>,
}

pub struct StatusCertificateAnalyzer {
    #[cfg(feature = "redis")]
    invalidation_publisher: Option<RedisStreamPublisher>,
}

impl StatusCertificateAnalyzer {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "redis")]
            invalidation_publisher: None,
        }
    }

    /// Attach Redis Streams publisher for AVM cache invalidation.
    #[cfg(feature = "redis")]
    pub fn with_invalidation_publisher(mut self, publisher: RedisStreamPublisher) -> Self {
        self.invalidation_publisher = Some(publisher);
        self
    }

    pub fn analyze(certificate_text: &str) -> StatusCertificateAnalysis {
        // existing analysis logic...
        let text_lower = certificate_text.to_lowercase();
        let mut warnings = vec![];
        let mut special_assessments_pending = false;
        let mut litigation_risk = false;

        if text_lower.contains("special assessment") {
            special_assessments_pending = true;
            warnings.push("Special Assessment detected".to_string());
        }
        if text_lower.contains("litigation") {
            litigation_risk = true;
            warnings.push("Litigation risk detected".to_string());
        }

        let overall_risk_level = if special_assessments_pending || litigation_risk {
            "High"
        } else {
            "Low"
        };

        StatusCertificateAnalysis {
            corporation_name: "Parsed Corp".to_string(),
            reserve_fund_balance: 1_250_000.0,
            special_assessments_pending,
            litigation_risk,
            insurance_status: "Adequate".to_string(),
            rules_violations: vec![],
            overall_risk_level: overall_risk_level.to_string(),
            warnings,
        }
    }

    /// Publish invalidation if high-risk findings are present.
    #[cfg(feature = "redis")]
    pub fn maybe_publish_invalidation(&self, property_key: &str, analysis: &StatusCertificateAnalysis) {
        if analysis.special_assessments_pending || analysis.litigation_risk {
            if let Some(publisher) = &self.invalidation_publisher {
                let _ = publisher.publish(property_key, "status_certificate_high_risk");
            }
        }
    }
}
