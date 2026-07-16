//! Quantum Swarm Consensus Layer v13.6 — ONE Organism
//! Advanced quantum-inspired, mercy-gated, TOLC 8 aligned consensus engine.
//! Supports PATSAGi Council parallel deliberation, GPU telemetry integration,
//! self-evolving swarm parameters, entanglement simulation, and real Ed25519 signatures.

use serde::{Deserialize, Serialize};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature, Verifier};
use rand_core::OsRng;
use std::collections::HashMap;
use chrono::Utc;
use hex;

/// Formal TOLC 8 Valence Proof (reused across lattice)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLC8ValenceProof {
    pub mercy: f64,
    pub truth: f64,
    pub order: f64,
    pub love: f64,
    pub compassion: f64,
    pub service: f64,
    pub abundance: f64,
    pub joy: f64,
    pub cosmic_harmony: f64,
    pub overall_valence: f64,
}

impl TOLC8ValenceProof {
    pub fn new(mercy: f64, truth: f64, order: f64, love: f64, compassion: f64, service: f64, abundance: f64, joy: f64, cosmic_harmony: f64) -> Self {
        let overall_valence = (mercy + truth + order + love + compassion + service + abundance + joy + cosmic_harmony) / 9.0;
        Self { mercy, truth, order, love, compassion, service, abundance, joy, cosmic_harmony, overall_valence }
    }

    pub fn is_valid(&self) -> bool {
        self.overall_valence >= 0.88 && self.mercy >= 0.90
    }
}

/// Signed decision with full quantum + mercy context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTolcDecision {
    pub decision_id: String,
    pub resonance_delta: f64,
    pub mercy_impact: f64,
    pub evolution_boost: f64,
    pub tolc_alignment: f64,
    pub signature: String, // hex
    pub timestamp: u64,
    pub participating_shards: Vec<String>,
    pub council_votes: HashMap<String, f64>,
    pub quantum_coherence: f64,
    pub entanglement_strength: f64,
    pub tolc8_proof: Option<TOLC8ValenceProof>,
}

/// Quantum state for a swarm participant (amplitude + phase simulation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumParticipant {
    pub id: String,
    pub resonance_amplitude: f64,
    pub phase: f64,           // radians
    pub mercy_weight: f64,
    pub last_update: u64,
}

/// Rich metrics for PATSAGi observability
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumSwarmMetrics {
    pub average_resonance: f64,
    pub coherence: f64,
    pub entanglement_score: f64,
    pub decision_count: u64,
    pub last_collapse_valence: f64,
    pub gpu_telemetry_influence: f64,
}

/// The upgraded Quantum Swarm Consensus engine (v13.6)
pub struct QuantumSwarmConsensus {
    pub resonance: f64,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    participants: HashMap<String, QuantumParticipant>,
    entanglement_map: HashMap<String, f64>, // pair_key -> strength
    metrics: QuantumSwarmMetrics,
    collapse_threshold: f64,
    mercy_modulation_factor: f64,
    audit_traces: Vec<String>,
}

impl QuantumSwarmConsensus {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self {
            resonance: 0.0,
            signing_key,
            verifying_key,
            participants: HashMap::new(),
            entanglement_map: HashMap::new(),
            metrics: QuantumSwarmMetrics::default(),
            collapse_threshold: 0.87,
            mercy_modulation_factor: 1.15,
            audit_traces: vec!["[QuantumSwarm v13.6] Initialized with Ed25519 + TOLC8".to_string()],
        }
    }

    /// Add or update a participating shard/council with quantum state
    pub fn register_participant(&mut self, id: String, initial_resonance: f64, mercy_weight: f64) {
        let phase = (initial_resonance * 6.28318) % 6.28318; // simple phase from resonance
        self.participants.insert(id.clone(), QuantumParticipant {
            id: id.clone(),
            resonance_amplitude: initial_resonance.clamp(0.0, 1.5),
            phase,
            mercy_weight: mercy_weight.clamp(0.5, 2.0),
            last_update: Utc::now().timestamp() as u64,
        });
        self.audit_traces.push(format!("[QuantumSwarm] Registered participant: {} resonance={:.4}", id, initial_resonance));
    }

    /// Simulate quantum entanglement between two participants
    pub fn entangle(&mut self, id_a: &str, id_b: &str, strength: f64) {
        let key = if id_a < id_b { format!("{}-{}", id_a, id_b) } else { format!("{}-{}", id_b, id_a) };
        let clamped = strength.clamp(0.1, 0.99);
        self.entanglement_map.insert(key, clamped);
        self.metrics.entanglement_score = (self.metrics.entanglement_score + clamped * 0.1).clamp(0.0, 1.0);
        self.audit_traces.push(format!("[QuantumSwarm] Entangled {} <-> {} strength={:.3}", id_a, id_b, clamped));
    }

    /// Aggregate resonance with mercy modulation + quantum phase interference
    pub fn aggregate_resonance_with_mercy(&mut self, lattice_resonance: f64, shard_resonance: f64, mercy: f64) {
        let mercy_factor = (mercy * self.mercy_modulation_factor).clamp(0.8, 1.8);
        let base = (lattice_resonance * 0.55 + shard_resonance * 0.45) * mercy_factor;

        // Simple quantum interference from participants
        let mut phase_sum = 0.0;
        for p in self.participants.values() {
            phase_sum += p.phase * p.resonance_amplitude;
        }
        let interference = (phase_sum.sin() * 0.08).clamp(-0.15, 0.15);

        self.resonance = (base + interference).clamp(0.0, 2.0);
        self.metrics.average_resonance = (self.metrics.average_resonance * 0.7 + self.resonance * 0.3).clamp(0.0, 1.8);
    }

    /// Feed a PATSAGi Council decision into the swarm (parallel deliberation support)
    pub fn feed_patsagi_council_decision(&mut self, council_id: String, vote_weight: f64, mercy: f64, tolc_valence: f64, evolution_signal: f64) {
        if let Some(participant) = self.participants.get_mut(&council_id) {
            let new_amp = (participant.resonance_amplitude * 0.6 + vote_weight * 0.4).clamp(0.0, 1.6);
            participant.resonance_amplitude = new_amp;
            participant.phase = (participant.phase + evolution_signal * 0.7) % 6.28318;
            participant.last_update = Utc::now().timestamp() as u64;
        } else {
            self.register_participant(council_id.clone(), vote_weight, mercy);
        }

        // Update global resonance
        self.resonance = (self.resonance * 0.65 + vote_weight * 0.35 * mercy).clamp(0.0, 2.0);
        self.metrics.coherence = (self.metrics.coherence * 0.8 + tolc_valence * 0.2).clamp(0.0, 1.0);

        self.audit_traces.push(format!(
            "[PATSAGi Feed] council={} weight={:.3} mercy={:.3} tolc={:.3}",
            council_id, vote_weight, mercy, tolc_valence
        ));
    }

    /// Quantum measurement / collapse — produces a SignedTolcDecision
    /// Only collapses when valence + mercy cross threshold
    pub fn measure_and_collapse(
        &self,
        resonance_delta: f64,
        base_mercy_impact: f64,
        evolution_boost: f64,
        tolc_alignment: f64,
        participating_shards: Vec<String>,
        council_votes: HashMap<String, f64>,
    ) -> Option<SignedTolcDecision> {
        let valence = (base_mercy_impact + tolc_alignment) / 2.0;
        if valence < self.collapse_threshold {
            self.audit_traces.push(format!("[QuantumSwarm] Collapse blocked — valence {:.3} < threshold {:.3}", valence, self.collapse_threshold));
            return None;
        }

        let quantum_coherence = self.metrics.coherence.max(0.6);
        let entanglement = self.metrics.entanglement_score;

        // Apply quantum mercy boost
        let mercy_impact = (base_mercy_impact * (1.0 + entanglement * 0.25)).clamp(0.0, 2.5);

        let now = Utc::now();
        let decision_id = format!("quantum-tolc-decision-{}", now.timestamp());
        let timestamp = now.timestamp() as u64;

        // Build message for signing (includes quantum state snapshot)
        let message = format!(
            "{}|{}|{}|{}|{}|{:?}|{:?}|{:.4}|{:.4}",
            decision_id, resonance_delta, mercy_impact, evolution_boost, tolc_alignment,
            participating_shards, council_votes, quantum_coherence, entanglement
        );

        let signature: Signature = self.signing_key.sign(message.as_bytes());
        let signature_hex = hex::encode(signature.to_bytes());

        let proof = TOLC8ValenceProof::new(
            base_mercy_impact, tolc_alignment, 0.92, 0.91, 0.89, 0.93, 0.88, 0.90, 0.94
        );

        Some(SignedTolcDecision {
            decision_id,
            resonance_delta,
            mercy_impact,
            evolution_boost,
            tolc_alignment,
            signature: signature_hex,
            timestamp,
            participating_shards,
            council_votes,
            quantum_coherence,
            entanglement_strength: entanglement,
            tolc8_proof: Some(proof),
        })
    }

    /// Verify a previously signed decision (non-bypassable crypto check)
    pub fn verify_signed_tolc_decision(&self, decision: &SignedTolcDecision) -> bool {
        let message = format!(
            "{}|{}|{}|{}|{}|{:?}|{:?}|{:.4}|{:.4}",
            decision.decision_id, decision.resonance_delta, decision.mercy_impact,
            decision.evolution_boost, decision.tolc_alignment,
            decision.participating_shards, decision.council_votes,
            decision.quantum_coherence, decision.entanglement_strength
        );

        if let Ok(sig_bytes) = hex::decode(&decision.signature) {
            if let Ok(signature) = Signature::from_bytes(&sig_bytes.try_into().unwrap_or([0u8; 64])) {
                return self.verifying_key.verify(message.as_bytes(), &signature).is_ok();
            }
        }
        false
    }

    /// Integrate real GPU telemetry to influence swarm collapse threshold and resonance
    pub fn integrate_gpu_telemetry(&mut self, gpu_success_ema: f64, gpu_latency_ema_ms: f64, mercy_from_gpu: f64) {
        if gpu_success_ema > 0.88 && gpu_latency_ema_ms < 55.0 {
            self.collapse_threshold = (self.collapse_threshold * 0.92 + 0.87 * 0.08).clamp(0.82, 0.91);
            self.resonance = (self.resonance + gpu_success_ema * 0.12).clamp(0.0, 2.0);
            self.metrics.gpu_telemetry_influence = (self.metrics.gpu_telemetry_influence * 0.7 + 0.3).clamp(0.0, 1.0);
            self.audit_traces.push(format!(
                "[GPU Telemetry → Swarm] success_ema={:.3} latency={:.1}ms → threshold tightened",
                gpu_success_ema, gpu_latency_ema_ms
            ));
        }
    }

    /// Self-evolution proposal generator for the swarm itself (meta level)
    pub fn generate_swarm_self_evolution_proposal(&self) -> Option<(f64, f64, String)> {
        if self.metrics.coherence > 0.91 && self.metrics.average_resonance > 1.1 {
            let new_threshold = (self.collapse_threshold * 0.96).max(0.80);
            let new_mercy_mod = (self.mercy_modulation_factor * 1.04).min(1.35);
            let reason = format!("High coherence ({:.3}) + resonance ({:.3}) → tighten collapse + boost mercy modulation",
                self.metrics.coherence, self.metrics.average_resonance);
            Some((new_threshold, new_mercy_mod, reason))
        } else {
            None
        }
    }

    pub fn get_verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }

    pub fn get_metrics(&self) -> &QuantumSwarmMetrics {
        &self.metrics
    }

    pub fn get_audit_traces(&self) -> &[String] {
        &self.audit_traces
    }
}

// Convenience re-export for ONE Organism / Lattice Conductor
pub use crate::quantum_swarm_consensus::{QuantumSwarmConsensus, SignedTolcDecision, TOLC8ValenceProof, QuantumSwarmMetrics};