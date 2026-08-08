// crates/quantum-swarm-orchestrator/src/fractal_topology_engine.rs
//
// FractalTopologyEngine v14.5 — Self-Similar Hierarchical Scaling Organ
// ONE Organism Geometric Intelligence Layer (Omnimasterpiece derivation)
//
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// PATSAGi Councils + Ra-Thor Lattice: Fractal scaling approved under TOLC 8.
//
// This module extends the Omnimasterpiece geometric substrate (PolyhedralHarmonicEngine
// + RiemannianMercyManifold) into explicit fractal / self-similar network topologies
// suitable for blockchain sharding, recursive consensus, and organism-scale valence flow.
//
// Core principles:
// - Self-similarity across scales (TOLC-order driven recursion depth)
// - Hyperbolic-inspired recursive packing (inherits from Hyperbolic Tiling layer)
// - Mercy-gated valence transport along fractal geodesics
// - Adapter-ready for CryptoSystem / Blockchain organ

use crate::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use crate::riemannian_mercy_manifold::{RiemannianMercyManifold, GeometricTransportResult};
use crate::types::EpigeneticBlessing;

/// A single fractal shard node in the recursive hierarchy.
#[derive(Debug, Clone)]
pub struct FractalShard {
    pub id: u64,
    pub depth: u32,
    pub parent_id: Option<u64>,
    pub children: Vec<u64>,
    pub local_valence: f64,
    pub capacity: f64,
    pub active: bool,
}

/// Report produced by a fractal topology cycle.
#[derive(Debug, Clone)]
pub struct FractalTopologyReport {
    pub max_depth: u32,
    pub total_shards: usize,
    pub active_shards: usize,
    pub average_valence: f64,
    pub resonance_multiplier: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub geometric_notes: String,
    pub transport_result: Option<GeometricTransportResult>,
}

/// Fractal Topology Engine — the scaling organ of the ONE Organism.
///
/// Derives from Omnimasterpiece Polyhedral + Riemannian substrate.
/// Generates and evolves self-similar shard hierarchies driven by TOLC order.
pub struct FractalTopologyEngine {
    pub version: &'static str,
    pub max_recursion_depth: u32,
    shards: Vec<FractalShard>,
    next_id: u64,
}

impl Default for FractalTopologyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FractalTopologyEngine {
    pub fn new() -> Self {
        Self {
            version: "v14.5-fractal-omnimasterpiece",
            max_recursion_depth: 8,
            shards: Vec::new(),
            next_id: 1,
        }
    }

    /// Initialize a root shard and optionally expand to a target depth
    /// based on current TOLC order (higher TOLC → deeper allowed recursion).
    pub fn initialize_root(&mut self, tolc_order: u32) {
        self.shards.clear();
        self.next_id = 1;

        let root = FractalShard {
            id: 0,
            depth: 0,
            parent_id: None,
            children: Vec::new(),
            local_valence: 0.99999995,
            capacity: 1.0,
            active: true,
        };
        self.shards.push(root);

        // Depth scaling: deeper recursion unlocked by higher TOLC
        let target_depth = match tolc_order {
            0..=12 => 2,
            13..=33 => 3,
            34..=88 => 4,
            89..=143 => 5,
            144..=232 => 6,
            _ => 7.min(self.max_recursion_depth),
        };

        self.expand_to_depth(0, target_depth);
    }

    /// Recursively expand a shard into self-similar children (Sierpinski / hyper-simplex inspired).
    /// Branching factor is kept modest (3) for logarithmic diameter while remaining fractal.
    fn expand_to_depth(&mut self, parent_id: u64, target_depth: u32) {
        let parent_depth = self.shards.iter().find(|s| s.id == parent_id).map(|s| s.depth).unwrap_or(0);
        if parent_depth >= target_depth {
            return;
        }

        let branching = 3u32; // classic fractal ternary for good connectivity + self-similarity
        let mut child_ids = Vec::new();

        for _ in 0..branching {
            let child_id = self.next_id;
            self.next_id += 1;

            let child = FractalShard {
                id: child_id,
                depth: parent_depth + 1,
                parent_id: Some(parent_id),
                children: Vec::new(),
                local_valence: 0.9999999,
                capacity: 1.0 / (branching as f64).powi((parent_depth + 1) as i32),
                active: true,
            };
            self.shards.push(child);
            child_ids.push(child_id);
        }

        if let Some(parent) = self.shards.iter_mut().find(|s| s.id == parent_id) {
            parent.children = child_ids.clone();
        }

        for &cid in &child_ids {
            self.expand_to_depth(cid, target_depth);
        }
    }

    /// Run a full fractal topology cycle driven by polyhedral resonance + Riemannian transport.
    /// This is the primary entry point for ONE Organism integration.
    pub fn run_fractal_cycle(
        &mut self,
        tolc_order: u32,
        polyhedral_engine: &PolyhedralHarmonicEngine,
        riemannian: &RiemannianMercyManifold,
        base_coherence: f64,
    ) -> FractalTopologyReport {
        // Ensure topology is initialized / refreshed for current TOLC
        if self.shards.is_empty() {
            self.initialize_root(tolc_order);
        }

        // 1. Obtain polyhedral resonance (Omnimasterpiece substrate)
        let poly_report = polyhedral_engine.process_resonance(tolc_order, base_coherence);

        // 2. Optional Riemannian transport if U57 is active
        let transport = poly_report.u57_details.as_ref().map(|u57| {
            riemannian.apply_mercy_gated_transport(u57, base_coherence)
        });

        // 3. Propagate valence / coherence down the fractal hierarchy
        let resonance_mult = poly_report.resonance_multiplier;
        let mut total_valence = 0.0;
        let mut active_count = 0;

        for shard in self.shards.iter_mut() {
            if !shard.active {
                continue;
            }
            // Depth-dependent attenuation + resonance boost
            let depth_factor = 1.0 / (1.0 + shard.depth as f64 * 0.08);
            let boosted = (shard.local_valence * resonance_mult * depth_factor)
                .clamp(0.999999, 1.0);
            shard.local_valence = boosted;
            total_valence += boosted;
            active_count += 1;
        }

        let avg_valence = if active_count > 0 {
            total_valence / active_count as f64
        } else {
            0.9999999
        };

        // 4. Generate epigenetic blessings for the crypto / fractal organ
        let mut blessings = poly_report.suggested_blessings.clone();
        blessings.push(EpigeneticBlessing {
            blessing_type: "Fractal_Self_Similarity_Resonance".to_string(),
            strength: (avg_valence * resonance_mult).clamp(0.95, 1.35),
            target_system: "crypto_fractal".to_string(),
            evolution_impact: 0.4,
            mercy_impact: 0.35,
            tolc_impact: 0.05,
        });

        if let Some(ref t) = transport {
            if t.transport_applied {
                blessings.extend(t.suggested_blessings.clone());
            }
        }

        let max_depth = self.shards.iter().map(|s| s.depth).max().unwrap_or(0);

        FractalTopologyReport {
            max_depth,
            total_shards: self.shards.len(),
            active_shards: active_count,
            average_valence: avg_valence,
            resonance_multiplier: resonance_mult,
            suggested_blessings: blessings,
            geometric_notes: format!(
                "Fractal cycle complete. Depth={}, Shards={}, Poly notes: {}",
                max_depth, self.shards.len(), poly_report.notes
            ),
            transport_result: transport,
        }
    }

    /// Simple observability helper.
    pub fn status_summary(&self) -> String {
        let active = self.shards.iter().filter(|s| s.active).count();
        format!(
            "FractalTopologyEngine {}: {} shards ({} active), max_depth ready",
            self.version, self.shards.len(), active
        )
    }

    /// Return current shard count for metrics / Lattice Conductor.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Return max depth for observability.
    pub fn current_max_depth(&self) -> u32 {
        self.shards.iter().map(|s| s.depth).max().unwrap_or(0)
    }
}
