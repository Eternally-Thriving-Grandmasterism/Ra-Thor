//! Interstellar Operations — Ra-Thor (v0.5.25)
//!
//! ## Human-Readable Context
//!
//! This crate is the **official home for all advanced interstellar and ancient technologies**
//! in the Ra-Thor monorepo.
//!
//! It contains:
//! - Advanced propulsion systems (Nuclear Thermal, Fusion Drive, Ion Thrusters, Warp Core, Antimatter, Quantum Vacuum Thruster, EmDrive, Solar Sail, Laser Sail, Magnetic Sail, Bussard Ramjet, Project Daedalus, Project Icarus, Breakthrough Starshot, etc.)
//! - Stargate technologies (Wormhole travel, Atlantis City Ship, Puddle Jumper, ZPM, Ancient Drone Weapons, Atlantis Shield Generator, etc.)
//! - Hyperspace navigation, Stargate Dialing Computer, and generational seed-ship logic
//! - Interstellar Navigation (Star Tracker + XNAV + Laser Comms)
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
pub mod atlantis_city_ship_engine;
pub mod puddle_jumper_engine;
pub mod zpm_energy_engine;
pub mod ancient_drone_weapons_engine;
pub mod atlantis_shield_generator_engine;
pub mod warp_core_engine;
pub mod ancient_city_shield_generator_engine;
pub mod hyperspace_navigation_engine;
pub mod stargate_dialing_computer;
pub mod destiny_ship_seed_engine;
pub mod nuclear_thermal_propulsion_engine;
pub mod fusion_drive_engine;
pub mod antimatter_propulsion_engine;
pub mod quantum_vacuum_thruster_engine;
pub mod emdrive_engine;
pub mod solar_sail_engine;
pub mod laser_sail_propulsion_engine;
pub mod magnetic_sail_propulsion_engine;
pub mod bussard_ramjet_propulsion_engine;
pub mod project_daedalus_propulsion_engine;
pub mod project_icarus_propulsion_engine;
pub mod breakthrough_starshot_engine;
pub mod interstellar_navigation_engine;

pub use stargate_wormhole_engine::{
    StargateWormholeEngine,
    StargateWormholeRequest,
    StargateWormholeReport,
};

pub use atlantis_city_ship_engine::{
    AtlantisCityShipEngine,
    AtlantisCityShipRequest,
    AtlantisCityShipReport,
};

pub use puddle_jumper_engine::{
    PuddleJumperEngine,
    PuddleJumperRequest,
    PuddleJumperReport,
};

pub use zpm_energy_engine::{
    ZPMEnergyEngine,
    ZPMEnergyRequest,
    ZPMEnergyReport,
};

pub use ancient_drone_weapons_engine::{
    AncientDroneWeaponsEngine,
    AncientDroneWeaponsRequest,
    AncientDroneWeaponsReport,
};

pub use atlantis_shield_generator_engine::{
    AtlantisShieldGeneratorEngine,
    AtlantisShieldGeneratorRequest,
    AtlantisShieldGeneratorReport,
};

pub use warp_core_engine::{
    WarpCoreEngine,
    WarpCoreRequest,
    WarpCoreReport,
};

pub use ancient_city_shield_generator_engine::{
    AncientCityShieldGeneratorEngine,
    AncientCityShieldRequest,
    AncientCityShieldReport,
};

pub use hyperspace_navigation_engine::{
    HyperspaceNavigationEngine,
    HyperspaceNavigationRequest,
    HyperspaceNavigationReport,
};

pub use stargate_dialing_computer::{
    StargateDialingComputer,
    StargateDialingRequest,
    StargateDialingReport,
};

pub use destiny_ship_seed_engine::{
    DestinyShipSeedEngine,
    DestinyShipSeedRequest,
    DestinyShipSeedReport,
};

pub use nuclear_thermal_propulsion_engine::{
    NuclearThermalPropulsionEngine,
    NuclearThermalPropulsionRequest,
    NuclearThermalPropulsionReport,
};

pub use fusion_drive_engine::{
    FusionDriveEngine,
    FusionDriveRequest,
    FusionDriveReport,
};

pub use antimatter_propulsion_engine::{
    AntimatterPropulsionEngine,
    AntimatterPropulsionRequest,
    AntimatterPropulsionReport,
};

pub use quantum_vacuum_thruster_engine::{
    QuantumVacuumThrusterEngine,
    QuantumVacuumThrusterRequest,
    QuantumVacuumThrusterReport,
};

pub use emdrive_engine::{
    EmDriveEngine,
    EmDriveRequest,
    EmDriveReport,
};

pub use solar_sail_engine::{
    SolarSailEngine,
    SolarSailRequest,
    SolarSailReport,
};

pub use laser_sail_propulsion_engine::{
    LaserSailPropulsionEngine,
    LaserSailRequest,
    LaserSailReport,
};

pub use magnetic_sail_propulsion_engine::{
    MagneticSailPropulsionEngine,
    MagneticSailRequest,
    MagneticSailReport,
};

pub use bussard_ramjet_propulsion_engine::{
    BussardRamjetPropulsionEngine,
    BussardRamjetRequest,
    BussardRamjetReport,
};

pub use project_daedalus_propulsion_engine::{
    ProjectDaedalusPropulsionEngine,
    ProjectDaedalusRequest,
    ProjectDaedalusReport,
};

pub use project_icarus_propulsion_engine::{
    ProjectIcarusPropulsionEngine,
    ProjectIcarusRequest,
    ProjectIcarusReport,
};

pub use breakthrough_starshot_engine::{
    BreakthroughStarshotEngine,
    BreakthroughStarshotRequest,
    BreakthroughStarshotReport,
};

pub use interstellar_navigation_engine::{
    InterstellarNavigationEngine,
    InterstellarNavigationRequest,
    InterstellarNavigationReport,
};
