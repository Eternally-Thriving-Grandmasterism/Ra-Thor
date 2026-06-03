//! RREL ↔ Lattice Conductor Bridge v14.4
//!
//! High-level orchestration between RREL and the parallel `LatticeConductor`.
//!
//! This bridge provides both low-level control and high-level convenience methods
//! for processing Real Estate offers using the high-performance parallel engine
//! (Rayon + DashMap + Arc caching).
//!
//! ## Method Overview
//!
//! | Method | Description | Recommended For |
//! |--------|-------------|-----------------|
//! | `conduct_offers_parallel` | Pure parallel conduction | When you already have prepared offers |
//! | `conduct_offers_batch` | Alias for parallel (default) | **Most common usage** |
//! | `process_and_conduct_offers_parallel` | High-level: assembly + parallel | When you want one-call processing |
//! | `conduct_offers_sequential` | Sequential fallback | Small batches or debugging |
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use real_estate_lattice::rrel_lattice_conductor_bridge::RrelLatticeConductorBridge;
//! use lattice_conductor::RealEstateOffer;
//! use patsagi_councils::PatsagiCouncil;
//! use std::sync::Arc;
//!
//! let coordinator: Arc<dyn PatsagiCouncil> = ...;
//! let bridge = RrelLatticeConductorBridge::new(coordinator);
//!
//! let offers: Vec<RealEstateOffer> = vec![ /* ... */ ];
//!
//! // Recommended: Simple parallel batch
//! let results = bridge.conduct_offers_batch(offers.clone());
//!
//! // Explicit parallel
//! let results = bridge.conduct_offers_parallel(offers.clone());
//!
//! // High-level method (future assembler integration)
//! let results = bridge.process_and_conduct_offers_parallel(offers);
//!
//! // Sequential fallback
//! let results = bridge.conduct_offers_sequential(offers);
//! ```
//!
//! ## When to Use Parallel?
//!
//! - Batches of **500+ offers** → Strongly recommended
//! - High-throughput simulations or bulk processing → Use `conduct_offers_batch`
//! - Small batches (< 300) → `conduct_offers_sequential` may be simpler

use crate::rrel_brokerage_assembler::RrelBrokerageAssembler;
use lattice_conductor::{LatticeConductor, RealEstateOffer, ConductedOffer, ConductorError};
use mercy::traits::{MercyAligned, TOLC8Gate};
use patsagi_councils::PatsagiCouncil;
use std::sync::Arc;

pub struct RrelLatticeConductorBridge {
    assembler: RrelBrokerageAssembler,
    conductor: LatticeConductor,
    coordinator: Arc<dyn PatsagiCouncil>,
}

impl RrelLatticeConductorBridge {
    pub fn new(coordinator: Arc<dyn PatsagiCouncil>) -> Self {
        Self {
            assembler: RrelBrokerageAssembler::new(),
            conductor: LatticeConductor::new(),
            coordinator,
        }
    }

    /// Sequential batch conduction
    pub fn conduct_offers_sequential(
        &mut self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        offers
            .into_iter()
            .map(|offer| self.conductor.conduct_real_estate_offer(offer))
            .collect()
    }

    /// Parallel batch conduction (Rayon + DashMap + Arc caching)
    pub fn conduct_offers_parallel(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conductor.conduct_batch(offers)
    }

    /// Recommended default batch method.
    /// Uses the optimized parallel path.
    pub fn conduct_offers_batch(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conduct_offers_parallel(offers)
    }

    /// High-level method that combines RREL assembly concepts with parallel conduction.
    /// Currently delegates to the parallel conductor. Future versions may enrich offers
    /// using the internal assembler before conduction.
    pub fn process_and_conduct_offers_parallel(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conductor.conduct_batch(offers)
    }

    pub fn expose_state_to_conductor(&self) -> Result<String, String> {
        Ok("RREL state exposed to Lattice Conductor v14.4 with parallel support".to_string())
    }

    pub fn receive_evolution_directive(&self, directive: &str) -> Result<(), String> {
        Ok(())
    }
}

impl MercyAligned for RrelLatticeConductorBridge {
    fn check_mercy_gates(&self) -> Vec<TOLC8Gate> {
        vec![
            TOLC8Gate::Genesis,
            TOLC8Gate::Truth,
            TOLC8Gate::Evolution,
            TOLC8Gate::Harmony,
            TOLC8Gate::Sovereignty,
            TOLC8Gate::Infinite,
        ]
    }
}
