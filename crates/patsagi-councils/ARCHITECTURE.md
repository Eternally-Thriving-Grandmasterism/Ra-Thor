patsagi-councils v0.5.14
├── Cargo.toml                    (workspace-ready, feature flags, correct paths)
│
├── src/
│   ├── lib.rs                    (Core Council Layer)
│   │   ├── CouncilFocus (16 variants: JoyAmplification → HyperonLattice)
│   │   ├── PATSAGiCouncil (evaluate_proposal, get_profile)
│   │   ├── PatsagiCouncilCoordinator
│   │   │   ├── conduct_voting_round (quorum + mercy valence)
│   │   │   ├── debate_and_consensus (cross-council)
│   │   │   └── run_eternal_governance_cycle
│   │   └── Re-exports + Prelude + #[cfg(modular-mercy)] adapter
│   │
│   ├── world_governance.rs       (Living Heart — WorldGovernanceEngine)
│   │   ├── FactionHarmonyMatrix (boost/reduce/peace/war prevention)
│   │   ├── FactionEconomy (multipliers, quantum entanglement, inflation)
│   │   ├── QuantumMercyField (pulse + propagate)
│   │   ├── FactionAIDiplomacy (treaties with duration/renew/break, alliances)
│   │   ├── FactionEspionage (intel, counter-intel, operations, mercy risk)
│   │   ├── FactionCulturalDynamics (festivals, heritage, exchange)
│   │   ├── FactionAIStrategyManager (8 strategies with scoring + execution)
│   │   └── WorldGovernanceEngine
│   │       ├── propose_and_approve_world_change (quantum swarm + mercy gate)
│   │       ├── run_full_world_cycle (decay + pulse + diplomacy + espionage + culture)
│   │       └── apply_world_impact (real PowrushGame mutations)
│   │
│   ├── mercy_engine_adapter.rs   (Feature-gated Mercy unification)
│   │
│   └── bin/
│       └── council_simulator.rs  (Interactive CLI tester)
│
└── (optional: examples/, tests/, README.md)
