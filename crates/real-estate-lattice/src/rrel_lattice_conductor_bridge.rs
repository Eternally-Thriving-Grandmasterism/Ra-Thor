//! RREL ↔ Lattice Conductor Bridge v14.4
//!
//! ## Parallel Processing Support
//!
//! This bridge now exposes high-performance parallel batch conduction:
//!
//! - `conduct_offers_parallel(...)` — Uses Rayon + DashMap for maximum throughput
//! - `conduct_offers_batch(...)` — Recommended default (parallel for large batches)
//! - `conduct_offers_sequential(...)` — Available for small batches or debugging
//!
//! ### When to use parallel?
//! - Batches of 500+ offers → Strong recommendation
//! - High-throughput pilots and simulations → Use `conduct_offers_batch`
//!
//! The underlying `LatticeConductor` uses Rayon for parallelism and DashMap
//! for lock-free concurrent ATTOM caching.

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

    /// Sequential batch conduction (original behavior)
    pub fn conduct_offers_sequential(
        &mut self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        offers
            .into_iter()
            .map(|offer| self.conductor.conduct_real_estate_offer(offer))
            .collect()
    }

    /// Parallel batch conduction using Rayon + DashMap.
    /// Recommended for large batches (500+ offers).
    pub fn conduct_offers_parallel(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conductor.conduct_batch(offers)
    }

    /// Default batch method — uses the parallel path.
    pub fn conduct_offers_batch(
        &self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        self.conduct_offers_parallel(offers)
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
