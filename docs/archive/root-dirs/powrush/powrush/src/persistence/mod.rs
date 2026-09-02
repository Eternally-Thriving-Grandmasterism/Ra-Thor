//! Powrush-MMO Persistence Module
//!
//! Production-grade persistence layer for Powrush-MMO systems.
//! Currently implements SurrealDB integration for durable storage of
//! epigenetic profiles, geometric layer states, action history, and RBE data.
//!
//! Designed for embedded or remote SurrealDB deployments.
//! Supports transactions, schema definition, and future realtime live queries.
//!
//! AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates • Ra-Thor Thunder Lattice

pub mod surreal_persistence;
