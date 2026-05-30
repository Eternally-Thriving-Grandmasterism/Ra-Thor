//! RREL Leptos Dashboard — Small Working View (v14.3)
//! Expanded into a functional dashboard view for the stabilized Real Estate Lattice.

use leptos::*;
use crate::{OntarioOfferFlowReport, OfferRiskSummary};

#[derive(Clone)]
pub struct DashboardState {
    pub recent_flows: Vec<OntarioOfferFlowReport>,
    pub total_offers: usize,
    pub valid_offers: usize,
    pub high_risk_offers: usize,
}

#[component]
pub fn RrelMainDashboard(state: DashboardState) -> impl IntoView {
    let valid_percentage = if state.total_offers > 0 {
        (state.valid_offers as f64 / state.total_offers as f64) * 100.0
    } else {
        0.0
    };

    view! {
        <div class="rrel-dashboard min-h-screen bg-zinc-950 text-white p-8">
            <div class="max-w-7xl mx-auto">
                <div class="flex items-center justify-between mb-8">
                    <div>
                        <h1 class="text-4xl font-bold tracking-tight">RREL Dashboard</h1>
                        <p class="text-zinc-400 mt-1">v14.3 • Ontario + USA Ready • Mercy-Gated</p>
                    </div>
                    <div class="px-4 py-2 bg-zinc-900 rounded-xl text-sm border border-zinc-800">
                        "TOLC 8 Active"
                    </div>
                </div>

                // Stats Row
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                    <StatCard title="Total Offers" value={state.total_offers.to_string()} />
                    <StatCard title="Valid Offers" value={state.valid_offers.to_string()} />
                    <StatCard title="High Risk" value={state.high_risk_offers.to_string()} />
                    <StatCard title="Validity Rate" value={format!("{:.0}%", valid_percentage)} />
                </div>

                // Recent Offer Flows
                <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
                    <div class="flex items-center justify-between mb-6">
                        <h2 class="text-xl font-semibold">Recent Offer Flows</h2>
                        <button class="px-4 py-2 text-sm bg-white text-black rounded-2xl font-medium hover:bg-zinc-200 transition-colors">
                            "Run New Flow"
                        </button>
                    </div>

                    <div class="space-y-3">
                        <For
                            each=move || state.recent_flows.clone()
                            key=|flow| flow.recommended_form.clone()
                            children=move |flow| {
                                let summary = OfferRiskSummary::from_flow_report(&flow);
                                view! {
                                    <div class="flex items-center justify-between bg-zinc-950 border border-zinc-800 rounded-2xl px-5 py-4 hover:border-zinc-700 transition-colors">
                                        <div class="flex items-center gap-x-4">
                                            <div class="font-mono text-sm text-zinc-400 w-28">{flow.deal_type.clone()}</div>
                                            <div class="font-medium">{flow.recommended_form.clone()}</div>
                                        </div>

                                        <div class="flex items-center gap-x-6 text-sm">
                                            <div class={if flow.offer_valid { "text-emerald-400" } else { "text-red-400" }}>
                                                {if flow.offer_valid { "Valid" } else { "Needs Review" }}
                                            </div>

                                            {if flow.multi_offer_escalation_triggered {
                                                view! { <div class="text-amber-400">Escalation</div> }
                                            } else { view! { <div></div> } }}

                                            <div class="text-zinc-400">{format!("Mercy {:.2}", flow.overall_mercy)}</div>
                                        </div>
                                    </div>
                                }
                            }
                        />
                    </div>
                </div>

                <div class="mt-6 text-xs text-zinc-500">
                    "All decisions mercy-gated • PATSAGi aligned • Quantum Swarm synced"
                </div>
            </div>
        </div>
    }
}

#[component]
pub fn StatCard(title: &'static str, value: String) -> impl IntoView {
    view! {
        <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-5">
            <div class="text-sm text-zinc-400">{title}</div>
            <div class="text-4xl font-semibold tracking-tighter mt-2">{value}</div>
        </div>
    }
}
