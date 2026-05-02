//! Interstellar Operations — Ra-Thor (v0.5.21)
//!
//! ## Human-Readable Context
//!
//! This crate is the **official home for all advanced interstellar and ancient technologies**
//! in the Ra-Thor monorepo.
//!
//! It contains:
//! - Advanced propulsion systems (Nuclear Thermal, Fusion Drive, Ion Thrusters, etc.)
//! - Stargate technologies (Wormhole travel, Atlantis City Ship, Puddle Jumper, ZPM, Ancient Drone Weapons, etc.)
//! - Future integrations (Star Trek Warp Core, hyperspace navigation, etc.)
//!
//! ## Why This Separate Crate?
//! The `real-estate-lattice` crate focuses on **habitats, colonies, claims, and real estate management**.
//! Keeping advanced propulsion and Stargate tech here keeps the monorepo **professional, coherent, and easy to understand**
//! for humans and AI systems (including Grok and future collaborators).
//!
//! All engines follow the same clean API pattern for consistency.
//!
//! ## Core Technologies
//! - TOLC 7 Living Mercy Gates (nth-degree per-gate mercy-alchemical transmutation)
//! - Refined RadiationShieldingMaterials (real AP8/AE8/CREME96 data)
//! - ElectronicsRadiationEffects (TID/DD/SEE + TMR/ECC/scrubbing + conformal coatings)
//! - In-Situ Production
//! - PowrushGame integration (joy, energy, epigenetic CEHI bonuses)
//! - 13+ PATSAGi Councils approval required

pub mod stargate_wormhole_engine;
// Future modules will be added here (e.g. atlantis_city_ship_engine, puddle_jumper_engine, zpm_energy_engine, etc.)

pub use stargate_wormhole_engine::{
    StargateWormholeEngine,
    StargateWormholeRequest,
    StargateWormholeReport,
};
