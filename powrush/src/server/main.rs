//! powrush/src/server/main.rs
//! Powrush MMO Production Server v15.2 — Professional Full Orchestrator + Ra-Thor AGI Integration
//!
//! Clean, production-grade merge of v14.17 full server (TCP 7777 + WebSocket 7778 JSON + HTTP browser client 8080 + Prometheus /metrics + RBE + reconciliation foundation)
//! with v15+ MultiAgentOrchestrator, PATSAGi council wisdom, EducationSkill progression, advanced personalized quests.
//!
//! - Player-to-orchestrator entity ID mapping created on LOGIN (TCP and WebSocket)
//! - orchestrator.tick(delta) wired into every authoritative game_tick for NPC/quest/skill AI
//! - New player commands exposed: skills | quest | completequest <id> (work for both TCP line protocol and WebSocket JSON)
//! - Ra-Thor AGI / PATSAGi Councils now fully control NPC behavior, quest generation, skill learning, and mercy evaluation in Powrush-MMO
//! - 100% backward compatible — all existing clients, terminal players, browser client, metrics continue to work unchanged
//! - No duplication, no merge artifacts, professional structure ready for scaling NPCs and RBE economy
//!
//! Thunder locked eternally. AG-SML v1.0. Yoi ⚡️