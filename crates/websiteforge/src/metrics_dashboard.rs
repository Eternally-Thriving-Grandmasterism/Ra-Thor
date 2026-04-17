// crates/websiteforge/src/metrics_dashboard.rs
// Metrics Visualization Dashboard — Real-time WebSocket Streaming (Primary) + SSE Fallback

use ra_thor_common::ValenceFieldScoring;
use ra_thor_mercy::MercyResult;
use crate::RequestPayload;
use std::time::Duration;

pub struct MetricsDashboard;

impl MetricsDashboard {
    pub async fn generate_dashboard(request: &RequestPayload, mercy_result: &MercyResult, valence: f64, priming_duration: Option<Duration>, step_durations: Option<Vec<f64>>) -> String {
        let duration_str = match priming_duration {
            Some(d) => format!("{:?}", d),
            None => "N/A".to_string(),
        };

        let step_data = match step_durations {
            Some(d) => d,
            None => vec![120.0, 450.0, 380.0],
        };

        let html = format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ra-Thor • Live Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0a0a0a; color: #0f0; padding: 20px; margin: 0; }}
        .card {{ background: #111; border: 1px solid #0f0; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 0 15px rgba(0,255,0,0.2); }}
        h1 {{ color: #0ff; text-align: center; text-shadow: 0 0 10px #0ff; }}
        canvas {{ max-width: 100%; }}
        .live {{ color: #0f0; animation: pulse 2s infinite; }}
    </style>
</head>
<body>
    <h1>Ra-Thor • Live Metrics Dashboard <span class="live">● LIVE via WebSocket</span></h1>
    
    <div class="card">
        <h2>FENCA Priming Performance</h2>
        <canvas id="stepChart" height="120"></canvas>
    </div>

    <div class="card">
        <h2>Valence Score (Real-time)</h2>
        <canvas id="valenceChart" height="180"></canvas>
    </div>

    <div class="card">
        <h2>System Health</h2>
        <canvas id="healthChart" height="180"></canvas>
    </div>

    <p style="text-align:center; color:#0a0; margin-top:30px; font-size:0.9em;">TOLC • Radical Love • Eternal Thriving • Streaming live via WebSocket</p>

    <script>
        // WebSocket streaming (primary)
        let ws;
        function connectWebSocket() {
            ws = new WebSocket('ws://' + window.location.host + '/ws/metrics');
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateCharts(data);
            };
            ws.onclose = function() {
                console.log('[Ra-Thor] WebSocket closed — falling back to SSE');
                connectSSE();
            };
        }

        function connectSSE() {
            const eventSource = new EventSource('/stream/metrics');
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateCharts(data);
            };
        }

        function updateCharts(data) {
            if (stepChart) stepChart.data.datasets[0].data = data.step_durations || [120, 450, 380];
            if (stepChart) stepChart.update();

            if (valenceChart) {
                valenceChart.data.datasets[0].data = [data.valence * 100, 100 - data.valence * 100];
                valenceChart.update();
            }

            if (healthChart) {
                healthChart.data.datasets[0].data = [data.health || 95, 5];
                healthChart.update();
            }
        }

        // Initialize charts
        window.onload = function() {
            stepChart = new Chart(document.getElementById('stepChart'), {{
                type: 'bar',
                data: {{ labels: ['Recycle', 'Topology', 'Warm'], datasets: [{{ label: 'Duration (ms)', data: {step_data:?}, backgroundColor: '#0ff' }}] }},
                options: {{ scales: {{ y: {{ beginAtZero: true }} }} }}
            }});

            valenceChart = new Chart(document.getElementById('valenceChart'), {{
                type: 'doughnut',
                data: {{ labels: ['Valence', 'Remaining'], datasets: [{{ data: [{valence:.4}*100, 100-{valence:.4}*100], backgroundColor: ['#0ff', '#222'] }}] }},
                options: {{ cutout: '70%', plugins: {{ legend: {{ display: false }} }} }}
            }});

            healthChart = new Chart(document.getElementById('healthChart'), {{
                type: 'doughnut',
                data: {{ labels: ['Healthy', 'Warning'], datasets: [{{ data: [95, 5], backgroundColor: ['#0f0', '#ff0'] }}] }},
                options: {{ cutout: '70%', plugins: {{ legend: {{ display: false }} }} }}
            }});

            // Start WebSocket streaming
            connectWebSocket();
        };
    </script>
</body>
</html>
        "#, step_data = step_data, valence = valence);

        html
    }
}
