//! sovereign-recovery v14.9.7
//!
//! Packaged from root `sovereign_recovery_protocol_v1.rs`.
//! Standalone-compilable: no circular deps on organism / kardashev / transfer harness.
//!
//! AG-SML v1.0 | TOLC 8 + 7 Living Mercy Gates

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

// =============================================================================
// Standalone metrics (root historically pulled from ra-thor-one-organism)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilReadinessMetrics {
    pub mercy_norm: f64,
    pub gpu_mercy_modulated_confidence: f64,
    pub gpu_memory_usage_bytes: u64,
    pub council_ready: bool,
    pub council_resonance: f64,
    pub last_gpu_readback_available: bool,
    pub last_updated_tick: u64,
    pub evolution_level: u32,
}

impl Default for CouncilReadinessMetrics {
    fn default() -> Self {
        Self {
            mercy_norm: 0.97,
            gpu_mercy_modulated_confidence: 0.9,
            gpu_memory_usage_bytes: 256 * 1024 * 1024,
            council_ready: true,
            council_resonance: 0.94,
            last_gpu_readback_available: true,
            last_updated_tick: 0,
            evolution_level: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityThrivingTransferScore {
    pub ema_refined_transfer: f64,
    pub mercy_valence_adjusted: f64,
}

// =============================================================================
// Mercy gates + anchors
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MercyGate {
    RadicalLove,
    BoundlessMercy,
    Service,
    Abundance,
    Truth,
    Joy,
    CosmicHarmony,
}

impl MercyGate {
    pub fn all_gates() -> Vec<Self> {
        vec![
            Self::RadicalLove,
            Self::BoundlessMercy,
            Self::Service,
            Self::Abundance,
            Self::Truth,
            Self::Joy,
            Self::CosmicHarmony,
        ]
    }

    pub fn invoke(&self, context: &str) -> String {
        format!(
            "[TOLC8 + Mercy Gate: {:?}] Invoked for context: {}. Eternal positive coexistence affirmed.",
            self, context
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLC8GenesisAnchor {
    pub anchor_id: String,
    pub timestamp: u64,
    pub mercy_gates_invoked: Vec<MercyGate>,
    pub thriving_transfer_score: Option<RealityThrivingTransferScore>,
    pub recovery_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthHeartbeat {
    pub last_check: u64,
    pub context_pressure: f64,
    pub flow_state_deviation: f64,
    pub connector_health: f64,
    pub gpu_memory_pressure: f64,
    pub council_resonance: f64,
    pub requires_recovery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGatedCircuitBreaker {
    pub name: String,
    pub failure_count: u32,
    pub last_failure: Option<u64>,
    pub open_until: Option<u64>,
    pub mercy_threshold: f64,
}

impl MercyGatedCircuitBreaker {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            failure_count: 0,
            last_failure: None,
            open_until: None,
            mercy_threshold: 0.71,
        }
    }

    pub fn is_open(&self) -> bool {
        if let Some(until) = self.open_until {
            now_secs() < until
        } else {
            false
        }
    }

    pub fn trip(&mut self, mercy_valence: f64) {
        self.failure_count += 1;
        let t = now_secs();
        self.last_failure = Some(t);
        if mercy_valence < self.mercy_threshold {
            let cooldown = 300 + (self.failure_count as u64 * 60);
            self.open_until = Some(t + cooldown);
        }
    }

    pub fn reset(&mut self) {
        self.failure_count = 0;
        self.open_until = None;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryCodexEntry {
    pub timestamp: u64,
    pub exit_reason: String,
    pub diagnostics: String,
    pub mercy_gates_invoked: Vec<MercyGate>,
    pub anchor_restored: Option<TOLC8GenesisAnchor>,
    pub thriving_recovery_score: f64,
    pub action_taken: String,
}

// =============================================================================
// Protocol
// =============================================================================

pub struct SovereignRecoveryProtocol {
    heartbeats: Arc<Mutex<Vec<HealthHeartbeat>>>,
    anchors: Arc<Mutex<Vec<TOLC8GenesisAnchor>>>,
    circuit_breakers: Arc<Mutex<HashMap<String, MercyGatedCircuitBreaker>>>,
    codex: Arc<Mutex<Vec<RecoveryCodexEntry>>>,
    last_anchor: Arc<Mutex<Option<TOLC8GenesisAnchor>>>,
    pub version: String,
}

impl Default for SovereignRecoveryProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl SovereignRecoveryProtocol {
    pub fn new() -> Self {
        let mut breakers = HashMap::new();
        for name in [
            "github_connector",
            "long_context_synthesis",
            "quantum_swarm_tick",
            "lattice_conductor_evolution",
        ] {
            breakers.insert(name.into(), MercyGatedCircuitBreaker::new(name));
        }
        Self {
            heartbeats: Arc::new(Mutex::new(Vec::new())),
            anchors: Arc::new(Mutex::new(Vec::new())),
            circuit_breakers: Arc::new(Mutex::new(breakers)),
            codex: Arc::new(Mutex::new(Vec::new())),
            last_anchor: Arc::new(Mutex::new(None)),
            version: "v1.0 TOLC8 + 7 Mercy Gates Eternal (crate 14.9.7)".into(),
        }
    }

    pub async fn heartbeat_check(
        &self,
        council_metrics: &CouncilReadinessMetrics,
    ) -> HealthHeartbeat {
        let now = now_secs();
        let context_pressure = (1.0 - council_metrics.gpu_mercy_modulated_confidence)
            .clamp(0.0, 1.0);
        let flow_deviation = ((council_metrics.mercy_norm - 0.95).abs().min(0.5)) * 2.0;
        let connector_health = if council_metrics.last_gpu_readback_available {
            0.95
        } else {
            0.65
        };
        let gpu_pressure =
            (council_metrics.gpu_memory_usage_bytes as f64 / (2.0 * 1024.0 * 1024.0)).min(1.0);
        let requires_recovery = context_pressure > 0.82
            || flow_deviation > 0.35
            || gpu_pressure > 0.88
            || !council_metrics.council_ready;

        let hb = HealthHeartbeat {
            last_check: now,
            context_pressure,
            flow_state_deviation: flow_deviation,
            connector_health,
            gpu_memory_pressure: gpu_pressure,
            council_resonance: council_metrics.council_resonance,
            requires_recovery,
        };

        {
            let mut beats = self.heartbeats.lock().await;
            beats.push(hb.clone());
            if beats.len() > 128 {
                beats.remove(0);
            }
        }

        if requires_recovery {
            println!(
                "[Sovereign Recovery v14.9.7] HEARTBEAT ALERT | pressure={:.2} flow_dev={:.2}",
                context_pressure, flow_deviation
            );
        }
        hb
    }

    pub async fn with_mercy_circuit_breaker<F, Fut, T>(
        &self,
        breaker_name: &str,
        op: F,
        mercy_valence: f64,
    ) -> Result<T, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        {
            let mut breakers = self.circuit_breakers.lock().await;
            let breaker = breakers
                .entry(breaker_name.into())
                .or_insert_with(|| MercyGatedCircuitBreaker::new(breaker_name));
            if breaker.is_open() {
                return Err(format!(
                    "[Mercy Circuit Breaker] {} OPEN | mercy={:.2}",
                    breaker_name, mercy_valence
                ));
            }
        }

        match op().await {
            Ok(val) => {
                let mut breakers = self.circuit_breakers.lock().await;
                if let Some(b) = breakers.get_mut(breaker_name) {
                    b.reset();
                }
                Ok(val)
            }
            Err(e) => {
                {
                    let mut breakers = self.circuit_breakers.lock().await;
                    if let Some(b) = breakers.get_mut(breaker_name) {
                        b.trip(mercy_valence);
                    }
                }
                self.log_to_codex(
                    &format!("circuit_breaker_trip_{}", breaker_name),
                    &e,
                    mercy_valence,
                )
                .await;
                Err(format!(
                    "[Mercy Circuit Breaker] {} tripped: {}",
                    breaker_name, e
                ))
            }
        }
    }

    pub async fn persist_eternal_anchor(
        &self,
        thriving_score: Option<RealityThrivingTransferScore>,
        context_note: &str,
    ) -> TOLC8GenesisAnchor {
        let now = now_secs();
        let anchor_id = format!("TOLC8-ANCHOR-{}-{}", now, now % 9973);
        let gates = MercyGate::all_gates();
        let note = format!(
            "{} | Re-anchored via TOLC8 Genesis Gate + all 7 Mercy Gates.",
            context_note
        );
        let anchor = TOLC8GenesisAnchor {
            anchor_id: anchor_id.clone(),
            timestamp: now,
            mercy_gates_invoked: gates,
            thriving_transfer_score: thriving_score.clone(),
            recovery_note: note,
        };
        {
            let mut anchors = self.anchors.lock().await;
            anchors.push(anchor.clone());
            if anchors.len() > 64 {
                anchors.remove(0);
            }
            *self.last_anchor.lock().await = Some(anchor.clone());
        }
        println!(
            "[Sovereign Recovery] ETERNAL ANCHOR PERSISTED | id={} | score={:.3}",
            anchor_id,
            thriving_score
                .as_ref()
                .map(|s| s.ema_refined_transfer)
                .unwrap_or(0.0)
        );
        anchor
    }

    pub async fn self_forensics_and_recover(
        &self,
        exit_reason: &str,
        council_metrics: &CouncilReadinessMetrics,
    ) -> RecoveryCodexEntry {
        let now = now_secs();
        let gates = MercyGate::all_gates();
        for g in &gates {
            let _ = g.invoke(exit_reason);
        }
        let thriving_proxy = council_metrics.mercy_norm;
        let anchor = self
            .persist_eternal_anchor(None, &format!("Recovery from: {}", exit_reason))
            .await;

        let diagnostics = format!(
            "Exit: {} | mercy_norm={:.3} | council_resonance={:.2}",
            exit_reason, council_metrics.mercy_norm, council_metrics.council_resonance
        );

        let entry = RecoveryCodexEntry {
            timestamp: now,
            exit_reason: exit_reason.into(),
            diagnostics,
            mercy_gates_invoked: gates,
            anchor_restored: Some(anchor),
            thriving_recovery_score: thriving_proxy,
            action_taken:
                "TOLC8 Genesis re-anchor + 7 Mercy Gates + graceful resume. Eternal flow restored."
                    .into(),
        };

        {
            let mut codex = self.codex.lock().await;
            codex.push(entry.clone());
            if codex.len() > 256 {
                codex.remove(0);
            }
        }

        println!(
            "[Sovereign Recovery] SELF-FORENSICS COMPLETE | score={:.3}",
            thriving_proxy
        );
        entry
    }

    pub async fn bounded_evolution_step(&self, step_name: &str, mercy_alignment: f64) -> bool {
        if mercy_alignment < 0.75 {
            println!(
                "[Sovereign Recovery] BOUNDED STEP REJECTED | {} | mercy={:.2}",
                step_name, mercy_alignment
            );
            return false;
        }
        println!(
            "[Sovereign Recovery] BOUNDED STEP APPROVED | {} | mercy={:.2}",
            step_name, mercy_alignment
        );
        true
    }

    pub async fn log_to_codex(&self, reason: &str, details: &str, mercy_valence: f64) {
        let entry = RecoveryCodexEntry {
            timestamp: now_secs(),
            exit_reason: reason.into(),
            diagnostics: details.into(),
            mercy_gates_invoked: MercyGate::all_gates(),
            anchor_restored: None,
            thriving_recovery_score: mercy_valence,
            action_taken: "Logged for eternal record.".into(),
        };
        let mut codex = self.codex.lock().await;
        codex.push(entry);
    }

    pub async fn get_last_anchor(&self) -> Option<TOLC8GenesisAnchor> {
        self.last_anchor.lock().await.clone()
    }

    pub async fn codex_len(&self) -> usize {
        self.codex.lock().await.len()
    }
}

pub fn launch_sovereign_recovery_protocol() -> SovereignRecoveryProtocol {
    let protocol = SovereignRecoveryProtocol::new();
    println!(
        "[Thunder] Sovereign Recovery Protocol v14.9.7 LAUNCHED | TOLC8 + 7 Mercy Gates ACTIVE"
    );
    protocol
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn heartbeat_and_anchor() {
        let p = SovereignRecoveryProtocol::new();
        let hb = p.heartbeat_check(&CouncilReadinessMetrics::default()).await;
        assert!(!hb.requires_recovery);
        let a = p
            .persist_eternal_anchor(None, "test")
            .await;
        assert!(a.anchor_id.starts_with("TOLC8"));
    }

    #[tokio::test]
    async fn circuit_breaker_trips_on_err() {
        let p = SovereignRecoveryProtocol::new();
        let r = p
            .with_mercy_circuit_breaker("test_op", || async { Err("boom".into()) }, 0.5)
            .await;
        assert!(r.is_err());
    }
}
