//! nevc_binding.rs
//! Explicit binding surface between Shared Valence Field and sealed NEVC scoring + lattice flow share
//! Phase B — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

use crate::{SharedValenceField, ValenceQuantum, Substrate};
use chrono::Utc;

/// Trait that any sealed NEVC scoring implementation must satisfy
pub trait NevcScoring {
    fn record_contribution(&mut self, quantum: &ValenceQuantum);
    fn current_collective_nevc(&self, instance_id: &str) -> f64;
}

/// Trait for lattice flow share broadcast (offline-first peer ingest)
pub trait LatticeFlowShare {
    fn broadcast_quantum(&self, quantum: &ValenceQuantum);
}

/// Binding helper that keeps the Shared Valence Field in sync with sealed NEVC + flow share
pub struct NevcFieldBinding<S, F>
where
    S: NevcScoring,
    F: LatticeFlowShare,
{
    pub scoring: S,
    pub flow: F,
}

impl<S, F> NevcFieldBinding<S, F>
where
    S: NevcScoring,
    F: LatticeFlowShare,
{
    pub fn new(scoring: S, flow: F) -> Self {
        Self { scoring, flow }
    }

    /// Emit through the field while simultaneously recording into NEVC and broadcasting
    pub fn emit_bound(&mut self, field: &mut SharedValenceField, quantum: ValenceQuantum) {
        // 1. Local field update (TOLC 8 floor already enforced inside emit)
        field.emit(quantum.clone());

        // 2. Record as fine-grained NEVC contribution event
        self.scoring.record_contribution(&quantum);

        // 3. Broadcast via lattice flow share for low-latency multi-player / multi-substrate propagation
        self.flow.broadcast_quantum(&quantum);
    }

    /// Convenience: emit a presence quantum (used by Symbiotic Membrane)
    pub fn emit_presence_bound(
        &mut self,
        field: &mut SharedValenceField,
        emitter_id: impl Into<String>,
        substrate: Substrate,
    ) {
        let quantum = SharedValenceField::emit_presence(emitter_id, substrate);
        self.emit_bound(field, quantum);
    }
}

/// Placeholder implementations for compilation until real NEVC + flow share crates are wired
#[derive(Default)]
pub struct PlaceholderNevcScoring;

impl NevcScoring for PlaceholderNevcScoring {
    fn record_contribution(&mut self, _quantum: &ValenceQuantum) {
        // Will call into sealed NEVC scoring surface
    }
    fn current_collective_nevc(&self, _instance_id: &str) -> f64 {
        0.999999
    }
}

#[derive(Default)]
pub struct PlaceholderLatticeFlowShare;

impl LatticeFlowShare for PlaceholderLatticeFlowShare {
    fn broadcast_quantum(&self, _quantum: &ValenceQuantum) {
        // Will call into sealed lattice flow share
    }
}
