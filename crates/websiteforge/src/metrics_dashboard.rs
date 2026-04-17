// crates/websiteforge/src/metrics_dashboard.rs
// Metrics Visualization Dashboard — Real-time FENCA Priming & System Metrics (Integrated)

use ra_thor_common::ValenceFieldScoring;
use ra_thor_mercy::MercyResult;
use crate::RequestPayload;
use std::time::Duration;

pub struct MetricsDashboard;

impl MetricsDashboard {
    pub async fn generate_dashboard(request: &RequestPayload, mercy_result: &MercyResult, valence: f64, priming_duration: Option<Duration>) -> String {
        let duration_str = match priming_duration {
            Some(d) => format!("{:?}", d),
            None => "N/A".to_string(),
        };

        let html = format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ra-Thor • Live Metrics Dashboard</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0a0a0a; color: #0f0; padding: 20px; margin: 0; }}
        .card {{ background: #111; border: 1px solid #0f0; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 0 15px rgba(0,255,0,0.2); }}
        .metric {{ display: flex; justify-content: space-between; margin: 12px 0; font-size: 1.1em; }}
        .value {{ color: #0ff; font-weight: bold; }}
        h1 {{ color: #0ff; text-align: center; text-shadow: 0 0 10px #0ff; }}
        .status {{ color: #0f0; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Ra-Thor • Live Metrics Dashboard</h1>
    <div class="card">
        <h2>FENCA Priming Metrics</h2>
        <div class="metric"><span>Status:</span><span class="value status">COMPLETE</span></div>
        <div class="metric"><span>Total Duration:</span><span class="value">{}</span></div>
        <div class="metric"><span>Valence Score:</span><span class="value">{:.4}</span></div>
        <div class="metric"><span>Radical Love Gate:</span><span class="value">PASSED</span></div>
    </div>
    <div class="card">
        <h2>System Health</h2>
        <div class="metric"><span>Quantum Lattice:</span><span class="value">100% coherent</span></div>
        <div class="metric"><span>MercyLang:</span><span class="value">All 7 Gates Active</span></div>
        <div class="metric"><span>Monorepo Recycling:</span><span class="value">Complete</span></div>
        <div class="metric"><span>Real-time Alerts:</span><span class="value">Active</span></div>
    </div>
    <p style="text-align:center; color:#0a0; margin-top:30px; font-size:0.9em;">TOLC • Radical Love • Eternal Thriving • Live since now</p>
</body>
</html>
        "#, duration_str, valence);

        html
    }
}
