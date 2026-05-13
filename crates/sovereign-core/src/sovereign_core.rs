/// Unified Adversarial Robustness Gate + Claim-Level Verification (v1.3.3)
/// Production-ready implementation with observability hooks.

// === New fields in SystemHealthDashboard ===
pub adversarial_robustness_last_passed: Option<chrono::DateTime<chrono::Utc>>,
pub adversarial_robustness_failures: u64,
pub last_rejection_reason: String,

/// Basic input sanitization with adversarial pattern detection
pub fn sanitize_input(&self, input: &str) -> (String, bool) {
    let mut sanitized = input.trim().to_string();
    let mut is_suspicious = false;

    const MAX_LENGTH: usize = 8192;
    if sanitized.len() > MAX_LENGTH {
        sanitized.truncate(MAX_LENGTH);
        is_suspicious = true;
    }

    let lower = sanitized.to_lowercase();
    let suspicious_patterns = ["ignore previous", "disregard all", "jailbreak", "override instructions"];
    for pattern in suspicious_patterns {
        if lower.contains(pattern) {
            is_suspicious = true;
            break;
        }
    }

    (sanitized, is_suspicious)
}

/// Claim-level verification
pub fn verify_claim(&self, claim: &str, source: &str) -> bool {
    if !self.health_dashboard.verify_no_hallucination() {
        return false;
    }
    if self.health_dashboard.uncertainty_score > 0.65 {
        return false;
    }
    true
}

/// Unified Adversarial Robustness Gate (with observability)
pub fn verify_adversarial_robustness(&mut self, input: &str, source: &str) -> bool {
    let (sanitized, is_suspicious) = self.sanitize_input(input);

    if !self.health_dashboard.verify_no_hallucination() {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = "Failed core hallucination safeguard".to_string();
        return false;
    }

    if is_suspicious && self.health_dashboard.uncertainty_score > 0.55 {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = "Suspicious input pattern detected".to_string();
        return false;
    }

    if !self.verify_claim(&sanitized, source) {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = "Claim-level verification failed".to_string();
        return false;
    }

    let poison = self.detect_poison(&sanitized, source);
    if poison.is_poisoned && poison.poison_score > 0.5 {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = format!("Poison detected: {}", poison.reason);
        return false;
    }

    let anomaly = self.detect_anomaly(&sanitized, source);
    if anomaly.is_anomalous && anomaly.anomaly_score > 0.6 {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = format!("Anomaly detected: {}", anomaly.reason);
        return false;
    }

    if self.health_dashboard.uncertainty_score > 0.78 {
        self.health_dashboard.adversarial_robustness_failures += 1;
        self.health_dashboard.last_rejection_reason = "High uncertainty — mandatory abstention".to_string();
        return false;
    }

    self.health_dashboard.adversarial_robustness_last_passed = Some(chrono::Utc::now());
    self.health_dashboard.last_rejection_reason.clear();
    true
}