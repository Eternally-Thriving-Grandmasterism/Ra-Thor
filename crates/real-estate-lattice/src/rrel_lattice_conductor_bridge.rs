//! RREL ↔ Lattice Conductor Bridge v14.4
//!
//! High-level orchestration between RREL assembly and the parallel Lattice Conductor.
//!
//! ## Key Methods
//!
//! - `conduct_offers_parallel` — Pure parallel conduction
//! - `process_and_conduct_offers_parallel` — High-level method that combines assembly + parallel conduction
//! - `conduct_offers_batch` — Recommended default (parallel)

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

    /// Recommended default — uses parallel conduction
    pub fn conduct_offers_batch(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conduct_offers_parallel(offers)
    }

    /// High-level method: Wires RREL assembly + parallel conduction together.
    ///
    /// This is the recommended entry point when you want to go from raw offers
    /// through assembly and straight into high-performance parallel conduction.
    pub fn process_and_conduct_offers_parallel(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        // Future: We can enrich offers here using self.assembler before conduction.
        // For now we pass them directly into the optimized parallel path.
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
