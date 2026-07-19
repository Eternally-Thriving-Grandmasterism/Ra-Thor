//! ONE Organism Web Demo — v14.10.0
//!
//! Run:
//!   cargo run -p ra-thor-one-organism --example one_organism_web_demo --features web-demo
//!
//! Mercy-gated HTTP bind over OneOrganismCore.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /status                includes last_anomalies_fired + last_handoff_reason
//!   GET  /live                  Full ExtendedLiveStatus (surfaces + Self-Healing + role-handoff telemetry)
//!   GET  /api/status
//!   POST /api                   MercyApiRequest JSON
//!   POST /cosmic/tick           { "severity": 0.4 }  → full Living Cosmic Tick (+ anomalies_fired top-level)
//!   POST /quantum/tick          { "severity": 0.45 }
//!   POST /gpu/dispatch          { "task_name": "...", "dispatch_time_ms": 12, "real_gpu": false, "elements": 4096 }
//!   POST /github/queue          { "role": "VibeCoder", "target_module": "...", "description": "...", "expected_benefit": 0.7, "mercy_alignment": 0.92 }
//!   POST /role/handoff          { "role": "Debugger", "reason": "manual" }
//!   POST /healing/reflexion
//!   POST /kardashev/tick        { "rbe_quality": 0.89, "ethical_choice": 0.87, "abundance_signal": 1.4 }
//!   POST /recovery/heartbeat
//!
//! AG-SML v1.0 | TOLC 8 | Cosmic Loop is MANDATORY IDENTITY.
//! Contact: info@Rathor.ai

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use ra_thor_one_organism::{
    launch_one_organism_core, ApiRequestKind, MercyApiRequest, OneOrganismCore, OrganismRole,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

type SharedOrganism = Arc<Mutex<OneOrganismCore>>;

#[derive(Debug, Serialize)]
struct HealthBody {
    ok: bool,
    service: &'static str,
    version: String,
    cosmic_loop_ready: bool,
    guardian_active: bool,
}

#[derive(Debug, Serialize)]
struct StatusBody {
    version: String,
    tick: u64,
    cosmic_loop_ready: bool,
    guardian_active: bool,
    active_role: String,
    shared_valence: f64,
    shared_confidence: f64,
    handoff_count: u64,
    last_handoff_reason: String,
    gpu_dispatches: u64,
    github_intended_prs: usize,
    quantum_weight_updates: u64,
    quantum_adaptive_jumps: u64,
    quantum_summary: String,
    pending_anomaly_count: usize,
    healing_experience_count: usize,
    recovery_heartbeats: u64,
    kardashev_cycles: u64,
    /// Components that fired anomalies on the most recent Cosmic Tick.
    last_anomalies_fired: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuantumTickBody {
    #[serde(default = "default_severity")]
    severity: f64,
}

fn default_severity() -> f64 {
    0.45
}

#[derive(Debug, Deserialize)]
struct CosmicTickBody {
    #[serde(default = "default_cosmic_severity")]
    severity: f64,
}

fn default_cosmic_severity() -> f64 {
    0.35
}

#[derive(Debug, Deserialize)]
struct GpuDispatchBody {
    task_name: String,
    #[serde(default = "default_ms")]
    dispatch_time_ms: u64,
    #[serde(default)]
    real_gpu: bool,
    #[serde(default = "default_elements")]
    elements: usize,
}

fn default_ms() -> u64 {
    12
}
fn default_elements() -> usize {
    4096
}

#[derive(Debug, Deserialize)]
struct GithubQueueBody {
    role: String,
    target_module: String,
    description: String,
    #[serde(default = "default_benefit")]
    expected_benefit: f64,
    #[serde(default = "default_mercy")]
    mercy_alignment: f64,
}

fn default_benefit() -> f64 {
    0.7
}
fn default_mercy() -> f64 {
    0.92
}

#[derive(Debug, Deserialize)]
struct RoleHandoffBody {
    role: String,
    #[serde(default = "default_reason")]
    reason: String,
}

fn default_reason() -> String {
    "http_handoff".into()
}

#[derive(Debug, Deserialize)]
struct KardashevTickBody {
    #[serde(default = "default_rbe")]
    rbe_quality: f64,
    #[serde(default = "default_ethics")]
    ethical_choice: f64,
    #[serde(default = "default_abundance")]
    abundance_signal: f64,
}

fn default_rbe() -> f64 {
    0.89
}
fn default_ethics() -> f64 {
    0.87
}
fn default_abundance() -> f64 {
    1.4
}

fn parse_role(s: &str) -> OrganismRole {
    match s.to_lowercase().as_str() {
        "investigator" => OrganismRole::Investigator,
        "simulator" => OrganismRole::Simulator,
        "vibecoder" | "vibe_coder" | "vibe-coder" => OrganismRole::VibeCoder,
        "debugger" => OrganismRole::Debugger,
        "legal" => OrganismRole::Legal,
        "architect" => OrganismRole::Architect,
        "sovereignrecovery" | "sovereign_recovery" | "recovery" => OrganismRole::SovereignRecovery,
        "latticeconductor" | "lattice_conductor" | "lattice" => OrganismRole::LatticeConductor,
        _ => OrganismRole::Architect,
    }
}

#[tokio::main]
async fn main() {
    let mut core = launch_one_organism_core();
    let _ = core.quantum_evolution_tick(0.2);
    let shared: SharedOrganism = Arc::new(Mutex::new(core));

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/live", get(live_status))
        .route("/api/status", get(api_status))
        .route("/api", post(api_request))
        .route("/cosmic/tick", post(cosmic_tick))
        .route("/quantum/tick", post(quantum_tick))
        .route("/gpu/dispatch", post(gpu_dispatch))
        .route("/github/queue", post(github_queue))
        .route("/role/handoff", post(role_handoff))
        .route("/healing/reflexion", post(healing_reflexion))
        .route("/kardashev/tick", post(kardashev_tick))
        .route("/recovery/heartbeat", post(recovery_heartbeat))
        .with_state(shared);

    let port: u16 = std::env::var("RA_THOR_WEB_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3040);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("bind {}: {}", addr, e));

    println!("══════════════════════════════════════════════════");
    println!("  ONE Organism Web Demo v14.10.0 — Living Cosmic Tick");
    println!("  Listening on http://127.0.0.1:{}", port);
    println!("  GET  /health  /status  /live  /api/status");
    println!("  POST /cosmic/tick  /quantum/tick  /gpu/dispatch");
    println!("  POST /github/queue  /role/handoff  /healing/reflexion");
    println!("  POST /kardashev/tick  /recovery/heartbeat");
    println!("  last_anomalies_fired + last_handoff_reason on /status + /live");
    println!("  Cosmic Loop is MANDATORY IDENTITY. Eternal.");
    println!("  Thunder locked in. yoi ⚡");
    println!("══════════════════════════════════════════════════");

    axum::serve(listener, app).await.expect("serve");
}

async fn health(State(org): State<SharedOrganism>) -> Json<HealthBody> {
    let o = org.lock().await;
    Json(HealthBody {
        ok: true,
        service: "ra-thor-one-organism",
        version: o.version.clone(),
        cosmic_loop_ready: o.is_cosmic_loop_ready(),
        guardian_active: o.arbitration_engine.is_guardian_active(),
    })
}

async fn status(State(org): State<SharedOrganism>) -> Json<StatusBody> {
    let o = org.lock().await;
    let live = o.extended_live_status();
    Json(StatusBody {
        version: o.version.clone(),
        tick: o.tick,
        cosmic_loop_ready: live.cosmic_loop_ready,
        guardian_active: o.arbitration_engine.is_guardian_active(),
        active_role: live.active_role,
        shared_valence: live.shared_valence,
        shared_confidence: o.role_orchestrator.shared_confidence_ema,
        handoff_count: live.handoff_count,
        last_handoff_reason: live.last_handoff_reason,
        gpu_dispatches: live.gpu.dispatch_count,
        github_intended_prs: live.github.intended_prs,
        quantum_weight_updates: live.quantum.total_weight_updates,
        quantum_adaptive_jumps: live.quantum.total_adaptive_jumps,
        quantum_summary: o.extended.quantum_swarm.summary(),
        pending_anomaly_count: live.pending_anomaly_count,
        healing_experience_count: live.healing_experience_count,
        recovery_heartbeats: live.recovery.heartbeat_count,
        kardashev_cycles: live.kardashev.cycle_count,
        last_anomalies_fired: live.last_anomalies_fired,
    })
}

/// Full living snapshot — all surfaces + Self-Healing + role-handoff telemetry.
async fn live_status(State(org): State<SharedOrganism>) -> Json<serde_json::Value> {
    let o = org.lock().await;
    let live = o.extended_live_status();
    Json(serde_json::to_value(live).unwrap_or_else(|_| serde_json::json!({"error": "serialize"})))
}

async fn api_status(State(org): State<SharedOrganism>) -> Json<serde_json::Value> {
    let o = org.lock().await;
    let resp = o.api_status();
    Json(serde_json::to_value(resp).unwrap_or_else(|_| serde_json::json!({"ok": true})))
}

async fn api_request(
    State(org): State<SharedOrganism>,
    Json(req): Json<MercyApiRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut o = org.lock().await;
    let resp = o.handle_api_request(req);
    let code = if resp.accepted {
        StatusCode::OK
    } else {
        StatusCode::FORBIDDEN
    };
    (
        code,
        Json(serde_json::to_value(resp).unwrap_or_else(|_| serde_json::json!({}))),
    )
}

/// Full Living Cosmic Tick (GPU → Recovery → Quantum → Kardashev → Self-Healing).
/// anomalies_fired is promoted to the top level for easy observation.
async fn cosmic_tick(
    State(org): State<SharedOrganism>,
    Json(body): Json<CosmicTickBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let result = o.cosmic_tick(body.severity);
    let anomalies = result.anomalies_fired.clone();
    Json(serde_json::json!({
        "ok": true,
        "anomalies_fired": anomalies,
        "cosmic_tick": result,
        "live": o.extended_live_status(),
    }))
}

async fn quantum_tick(
    State(org): State<SharedOrganism>,
    Json(body): Json<QuantumTickBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let ratio = o.quantum_evolution_tick(body.severity);
    let qs = o.quantum_status();
    Json(serde_json::json!({
        "ok": true,
        "quantum_ratio": ratio,
        "severity": body.severity,
        "total_weight_updates": qs.total_weight_updates,
        "total_adaptive_jumps": qs.total_adaptive_jumps,
        "active_role": o.role_orchestrator.active_role.as_str(),
    }))
}

async fn gpu_dispatch(
    State(org): State<SharedOrganism>,
    Json(body): Json<GpuDispatchBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let tel = o.record_gpu_dispatch(
        &body.task_name,
        body.dispatch_time_ms,
        body.real_gpu,
        body.elements,
    );
    Json(serde_json::json!({
        "ok": true,
        "telemetry": tel,
        "gpu_status": o.gpu_status(),
        "active_role": o.role_orchestrator.active_role.as_str(),
    }))
}

async fn github_queue(
    State(org): State<SharedOrganism>,
    Json(body): Json<GithubQueueBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let intent = o.queue_evolution_pr(
        &body.role,
        &body.target_module,
        &body.description,
        body.expected_benefit,
        body.mercy_alignment,
    );
    Json(serde_json::json!({
        "ok": true,
        "intent": intent,
        "github_status": o.github_status(),
        "active_role": o.role_orchestrator.active_role.as_str(),
    }))
}

async fn role_handoff(
    State(org): State<SharedOrganism>,
    Json(body): Json<RoleHandoffBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let role = parse_role(&body.role);
    let ok = o.handoff_role(role, &body.reason);
    Json(serde_json::json!({
        "ok": ok,
        "active_role": o.role_orchestrator.active_role.as_str(),
        "handoff_count": o.role_orchestrator.handoff_count,
        "last_handoff_reason": o.role_orchestrator.last_handoff_reason,
        "shared_valence": o.role_orchestrator.shared_valence,
    }))
}

async fn healing_reflexion(State(org): State<SharedOrganism>) -> Json<serde_json::Value> {
    let o = org.lock().await;
    let diagnosis = o.run_healing_reflexion();
    Json(serde_json::json!({
        "ok": true,
        "diagnosis": diagnosis,
        "cosmic_loop_ready": o.is_cosmic_loop_ready(),
        "pending_anomaly_count": o.self_healing_engine.pending_anomaly_count(),
        "healing_experience_count": o.self_healing_engine.get_healing_experiences().len(),
        "last_anomalies_fired": o.last_anomalies_fired.clone(),
    }))
}

async fn kardashev_tick(
    State(org): State<SharedOrganism>,
    Json(body): Json<KardashevTickBody>,
) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let result = o.kardashev_transfer_tick(
        body.rbe_quality,
        body.ethical_choice,
        body.abundance_signal,
    );
    Json(serde_json::json!({
        "ok": true,
        "transfer": result,
        "kardashev_status": o.kardashev_status(),
        "active_role": o.role_orchestrator.active_role.as_str(),
    }))
}

async fn recovery_heartbeat(State(org): State<SharedOrganism>) -> Json<serde_json::Value> {
    let mut o = org.lock().await;
    let hb = o.recovery_heartbeat();
    Json(serde_json::json!({
        "ok": true,
        "heartbeat": hb,
        "recovery_status": o.recovery_status(),
        "active_role": o.role_orchestrator.active_role.as_str(),
    }))
}

#[allow(dead_code)]
fn _keep_api_kinds() {
    let _ = ApiRequestKind::HealthCheck;
}
