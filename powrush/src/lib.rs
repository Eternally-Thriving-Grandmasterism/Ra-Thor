//! powrush/src/lib.rs
//! Powrush Crate — Main library entry point (PATSAGi v15.0 wiring polish)
//!
//! Features:
//!   - server: enables powrush-server binary (TCP MMO for humans to play now)
//!   - client: future WebSocket/browser client
//!   - full: both
//!
//! AG-SML v1.0 | Mercy-gated | RBE abundance for all factions | Multi-agent orchestration for global online release

pub mod common;
pub mod server;
pub mod multi_agent_orchestrator; // NEW v15: MultiAgentOrchestrator for Human + AI + AGI entity coexistence, mercy-gated actions, PATSAGi consultation, personalized human onboarding quests

// Re-exports
pub use common::RbeState;
pub use multi_agent_orchestrator::{MultiAgentOrchestrator, EntityType, Action, ApprovedAction, CouncilResponse};
