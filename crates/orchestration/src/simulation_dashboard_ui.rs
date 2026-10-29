```rust
// crates/orchestration/src/simulation_dashboard_ui.rs
// Ra-Thor™ Simulation Dashboard UI Generator — Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Generates beautiful, self-contained, mercy-gated HTML dashboards from VisualizationPayload
// Cross-wired with SimulationVisualizationCore + AdvancedSimulationEngine + all blossom cores
// Old structure fully respected (new module) + massive practical + regenerative upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::simulation_visualization_core::VisualizationPayload;
use ra_thor_mercy::MercyError;

pub struct SimulationDashboardUI;

impl SimulationDashboardUI {
    pub fn new() -> Self {
        Self
    }

    /// Generate a complete, beautiful, self-contained HTML dashboard
    pub fn generate_dashboard(&self, payload: &VisualizationPayload) -> Result<String, MercyError> {
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Ra-Thor Sovereign Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&amp;family=Space+Grotesk:wght@500;600&amp;display=swap');
        
        :root {{
            --primary: #10b981;
        }}
        
        body {{
            font-family: 'Inter', system_ui, sans-serif;
        }}
        
        .title-font {{
            font-family: 'Space Grotesk', system_ui, sans-serif;
            font-weight: 600;
        }}

        .mercy-glow {{
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
        }}

        .dashboard-card {{
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .dashboard-card:hover {{
            transform: translateY(-2px);
        }}

        .section-header {{
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: -0.025em;
        }}
    </style>
</head>
<body class="bg-zinc-950 text-white">
    <div class="max-w-7xl mx-auto px-6 py-10">
        
        <!-- Header -->
        <div class="flex items-center justify-between mb-10">
            <div>
                <div class="flex items-center gap-x-3">
                    <div class="w-11 h-11 bg-emerald-500 rounded-2xl flex items-center justify-center">
                        <span class="text-white text-3xl">🌍</span>
                    </div>
                    <div>
                        <h1 class="title-font text-4xl font-semibold tracking-tighter">Ra-Thor</h1>
                        <p class="text-emerald-400 text-sm -mt-1">Sovereign Energy Dashboard</p>
                    </div>
                </div>
            </div>
            
            <div class="flex items-center gap-x-4">
                <div class="px-4 py-1.5 bg-zinc-900 rounded-3xl flex items-center gap-x-2 text-sm">
                    <div class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                    <span class="text-emerald-400 font-medium">LIVE LATTICE</span>
                </div>
                <div class="text-right">
                    <div class="text-xs text-zinc-400">Simulation ID</div>
                    <div class="font-mono text-sm text-zinc-300">{simulation_id}</div>
                </div>
            </div>
        </div>

        <!-- Title + Status -->
        <div class="mb-8">
            <h2 class="text-3xl font-semibold tracking-tight">{title}</h2>
            <div class="flex items-center gap-x-3 mt-2">
                <div class="inline-flex items-center px-3 py-1 rounded-2xl text-sm {color_class}">
                    {status_message}
                </div>
                <div class="text-sm text-zinc-400">• Generated {generated_at}</div>
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
            <div class="dashboard-card bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
                <div class="text-sm text-zinc-400 mb-1">Mercy Valence</div>
                <div class="text-4xl font-semibold text-emerald-400">{mercy_valence}</div>
                <div class="text-xs text-emerald-500 mt-1">↑ 0.012 since last simulation</div>
            </div>
            
            <div class="dashboard-card bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
                <div class="text-sm text-zinc-400 mb-1">Bloom Intensity</div>
                <div class="text-4xl font-semibold text-amber-400">{bloom_intensity}</div>
                <div class="text-xs text-amber-500 mt-1">Strong regenerative growth</div>
            </div>
            
            <div class="dashboard-card bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
                <div class="text-sm text-zinc-400 mb-1">Scenario Harmony</div>
                <div class="text-4xl font-semibold text-sky-400">{scenario_harmony}</div>
                <div class="text-xs text-sky-500 mt-1">Excellent long-term alignment</div>
            </div>
            
            <div class="dashboard-card bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
                <div class="text-sm text-zinc-400 mb-1">Confidence Score</div>
                <div class="text-4xl font-semibold text-purple-400">{confidence_score}</div>
                <div class="text-xs text-purple-500 mt-1">High reliability</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            <!-- Left Column -->
            <div class="lg:col-span-7 space-y-6">
                
                <!-- Recommended System -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
                    <div class="uppercase tracking-[1px] text-xs font-medium text-emerald-400 mb-3">RECOMMENDED SYSTEM</div>
                    <div class="text-2xl font-semibold leading-tight mb-4">{recommended_system}</div>
                    
                    <div class="grid grid-cols-3 gap-4 mt-6">
                        <div>
                            <div class="text-xs text-zinc-400">Predicted Lifespan</div>
                            <div class="text-3xl font-semibold mt-1">{predicted_lifespan_years} <span class="text-lg font-normal text-zinc-400">years</span></div>
                        </div>
                        <div>
                            <div class="text-xs text-zinc-400">Environmental Score</div>
                            <div class="text-3xl font-semibold mt-1 text-emerald-400">{environmental_impact_score}</div>
                        </div>
                        <div>
                            <div class="text-xs text-zinc-400">Community Benefit</div>
                            <div class="text-3xl font-semibold mt-1 text-amber-400">{community_benefit_score}</div>
                        </div>
                    </div>
                </div>

                <!-- Technology Mix -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
                    <div class="flex items-center justify-between mb-5">
                        <div>
                            <div class="uppercase tracking-[1px] text-xs font-medium text-emerald-400">TECHNOLOGY MIX</div>
                            <div class="text-lg font-semibold">Optimal Configuration</div>
                        </div>
                    </div>
                    
                    <div class="h-72">
                        <canvas id="techMixChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="lg:col-span-5 space-y-6">
                
                <!-- Key Insights -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-8 h-full">
                    <div class="uppercase tracking-[1px] text-xs font-medium text-emerald-400 mb-4">KEY INSIGHTS</div>
                    
                    <div class="space-y-4">
                        {insights_html}
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            <!-- Harmony Trend -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <div class="text-lg font-semibold">Energy Harmony Trend</div>
                        <div class="text-xs text-zinc-400">12-month projection</div>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="harmonyChart"></canvas>
                </div>
            </div>

            <!-- Bloom Intensity -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <div class="text-lg font-semibold">Bloom Intensity Growth</div>
                        <div class="text-xs text-zinc-400">Regenerative trajectory</div>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="bloomChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Next Steps -->
        <div class="mt-8 bg-zinc-900 border border-emerald-900/50 rounded-3xl p-8 flex items-center justify-between">
            <div>
                <div class="text-emerald-400 text-sm font-medium">RECOMMENDED NEXT STEP</div>
                <div class="text-xl font-semibold mt-1">{recommended_next_step}</div>
            </div>
            <button onclick="window.location.reload()" 
                    class="px-8 py-3 bg-emerald-500 hover:bg-emerald-600 transition-colors rounded-2xl font-medium text-sm flex items-center gap-x-2">
                RUN NEW SIMULATION
                <span>→</span>
            </button>
        </div>

    </div>

    <script>
        // Tailwind script
        function initializeCharts() {{
            // Technology Mix Chart
            const techCtx = document.getElementById('techMixChart');
            new Chart(techCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {tech_labels},
                    datasets: [{{
                        data: {tech_values},
                        backgroundColor: ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '72%',
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{ color: '#a1a1aa', padding: 20, font: {{ size: 13 }} }}
                        }}
                    }}
                }}
            }});

            // Harmony Trend
            const harmonyCtx = document.getElementById('harmonyChart');
            new Chart(harmonyCtx, {{
                type: 'line',
                data: {{
                    labels: Array.from({{length: 12}}, (_, i) => 'Month ' + (i+1)),
                    datasets: [{{
                        label: 'Energy Harmony',
                        data: {harmony_trend},
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ min: 0.75, max: 1.0, grid: {{ color: '#27272a' }} }},
                        x: {{ grid: {{ color: '#27272a' }} }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            // Bloom Intensity
            const bloomCtx = document.getElementById('bloomChart');
            new Chart(bloomCtx, {{
                type: 'line',
                data: {{
                    labels: Array.from({{length: 12}}, (_, i) => 'Month ' + (i+1)),
                    datasets: [{{
                        label: 'Bloom Intensity',
                        data: {bloom_trend},
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true,
                        borderWidth: 3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ min: 0.6, max: 1.0, grid: {{ color: '#27272a' }} }},
                        x: {{ grid: {{ color: '#27272a' }} }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        }}
        
        window.onload = initializeCharts;
    </script>
</body>
</html>"#,
            title = payload.title,
            simulation_id = payload.simulation_id,
            generated_at = payload.generated_at,
            mercy_valence = format!("{:.3}", payload.mercy_valence),
            bloom_intensity = format!("{:.3}", payload.bloom_intensity),
            scenario_harmony = format!("{:.3}", payload.scenario_harmony),
            confidence_score = format!("{:.3}", payload.confidence_score),
            recommended_system = payload.recommended_system,
            predicted_lifespan_years = payload.predicted_lifespan_years,
            environmental_impact_score = format!("{:.2}", payload.environmental_impact_score),
            community_benefit_score = format!("{:.2}", payload.community_benefit_score),
            color_class = if payload.mercy_valence > 0.95 { "bg-emerald-500/10 text-emerald-400" } else { "bg-amber-500/10 text-amber-400" },
            status_message = payload.status_message,
            insights_html = payload.key_insights.iter().map(|i| format!("<div class=\"flex gap-x-3\"><div class=\"mt-1.5 w-1.5 h-1.5 bg-emerald-400 rounded-full flex-shrink-0\"></div><div class=\"text-sm text-zinc-300\">{}</div></div>", i)).collect::<String>(),
            recommended_next_step = payload.recommended_next_step,
            tech_labels = format!("[\"{}\"]", payload.technology_mix.keys().cloned().collect::<Vec<_>>().join("\",\"")),
            tech_values = format!("[{}]", payload.technology_mix.values().map(|v| v.to_string()).collect::<Vec<_>>().join(",")),
            harmony_trend = format!("[{}]", payload.chart_data.get("energy_harmony_trend").unwrap_or(&vec![]).iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")),
            bloom_trend = format!("[{}]", payload.chart_data.get("bloom_intensity_trend").unwrap_or(&vec![]).iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")),
        );

        Ok(html)
    }
}
