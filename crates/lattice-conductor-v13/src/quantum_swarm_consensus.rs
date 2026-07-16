//! Quantum Swarm Consensus Layer v13.7 — ONE Organism + v15.33 Geometric Intelligence Fusion
//! Advanced quantum-inspired, mercy-gated, TOLC 8 aligned consensus engine with **Geometric + Swarm Fusion** and **Harmony Caching Hooks**.
//! Supports PATSAGi Council parallel deliberation, GPU telemetry integration,
//! self-evolving swarm parameters, entanglement simulation, real Ed25519 signatures, and now deep entanglement with GeometricState (tolc_alignment, valence, mercy_score).

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

/// v15.33: Harmony Cache Entry — stores high-fused (GeometricState + Swarm) snapshots for acceleration
/// Used in repeated GeometricUpdate + SwarmConsensusDispatch passes to skip redundant computation when harmony is high.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyCacheEntry {
    pub geometric_tolc_alignment: f64,
    pub geometric_valence: f64,
    pub geometric_mercy: f64,
    pub swarm_coherence: f64,
    pub fused_harmony: f64,
    pub timestamp: u64,
}

/// The upgraded Quantum Swarm Consensus engine (v13.7 + v15.33 Geometric Fusion)
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
    // v15.33 Harmony Caching
    harmony_cache: HashMap<String, HarmonyCacheEntry>,
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
            audit_traces: vec!["[QuantumSwarm v13.7] Initialized with Ed25519 + TOLC8 + Geometric Fusion v15.33".to_string()],
            harmony_cache: HashMap::new(),
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

    // ============================================================
    // v15.33 DEEPER FUSION: Geometric Intelligence + Quantum Swarm Consensus
    // ============================================================

    /// Fuse GeometricState (from geometric.rs) influence into swarm resonance/coherence.
    /// Called from SelfEvolutionOrchestrator or ONE Organism after GeometricUpdate pass or GeometricMotor apply.
    /// Returns the fused harmony score for downstream use (e.g. boosting dispatch or evolution proposal).
    pub fn fuse_geometric_state(&mut self, geo_tolc_alignment: f64, geo_valence: f64, geo_mercy: f64, base_coherence: f64, mercy: f64) -> f64 {
        let geo_boost = (geo_tolc_alignment * 0.4 + geo_valence * 0.3 + geo_mercy * 0.3).clamp(0.0, 1.5);
        let fused = (base_coherence * 0.6 + geo_boost * 0.4) * mercy.clamp(0.8, 1.5);
        self.resonance = (self.resonance + fused * 0.1).clamp(0.0, 2.5);
        self.metrics.coherence = (self.metrics.coherence * 0.85 + fused * 0.15).clamp(0.0, 1.0);
        self.audit_traces.push(format!("[v15.33 Geometric+Swarm Fusion] geo_tolc={:.3} geo_valence={:.3} geo_mercy={:.3} base_coh={:.3} → fused_harmony={:.3}", geo_tolc_alignment, geo_valence, geo_mercy, base_coherence, fused));
        fused
    }

    /// Harmony Caching Hook v15.33 — store/retrieve high-fused snapshots for acceleration.
    /// In GPU pipeline GeometricUpdate + SwarmConsensusDispatch loops, this provides fast-path when recent harmony is high (reduces redundant entanglement/collapse computation).
    /// Returns (fused_harmony_score, was_cache_hit)
    pub fn cache_or_retrieve_harmony(&mut self, cache_key: &str, geo_tolc_alignment: f64, geo_valence: f64, geo_mercy: f64, current_coherence: f64, mercy: f64) -> (f64, bool) {
        let now = Utc::now().timestamp() as u64;
        if let Some(entry) = self.harmony_cache.get(cache_key) {
            if (now - entry.timestamp) < 5000 && entry.fused_harmony > 0.92 {  // 5s freshness window + high harmony threshold
                self.audit_traces.push(format!("[HarmonyCache HIT v15.33] key={} fused={:.3} age_ms={}", cache_key, entry.fused_harmony, now - entry.timestamp));
                return (entry.fused_harmony, true);
            }
        }
        // Miss or stale → compute fresh fusion and cache
        let fused = self.fuse_geometric_state(geo_tolc_alignment, geo_valence, geo_mercy, current_coherence, mercy);
        let entry = HarmonyCacheEntry {
            geometric_tolc_alignment: geo_tolc_alignment,
            geometric_valence: geo_valence,
            geometric_mercy: geo_mercy,
            swarm_coherence: current_coherence,
            fused_harmony: fused,
            timestamp: now,
        };
        self.harmony_cache.insert(cache_key.to_string(), entry);
        (fused, false)
    }

    /// Optional maintenance: clear stale harmony cache entries
    pub fn clear_stale_harmony_cache(&mut self, max_age_ms: u64) {
        let now = Utc::now().timestamp() as u64;
        self.harmony_cache.retain(|_, e| (now - e.timestamp) < max_age_ms);
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
pub use crate::quantum_swarm_consensus::{QuantumSwarmConsensus, SignedTolcDecision, TOLC8ValenceProof, QuantumSwarmMetrics, HarmonyCacheEntry};