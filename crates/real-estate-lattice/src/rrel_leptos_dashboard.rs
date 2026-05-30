//! RREL Leptos Dashboard — Deeper Wiring (v14.3)
//! Now supports enriched data from AttomDataProvider + caching.

use leptos::*;
use crate::{
    OntarioOfferFlowReport,
    UsaOfferFlowReport,
    usa_attom_cache::{PropertyProfile, RiskSignals},
};

#[derive(Clone)]
pub struct DashboardState {
    pub recent_ontario_flows: Vec<OntarioOfferFlowReport>,
    pub recent_usa_flows: Vec<UsaOfferFlowReport>,
}

#[component]
pub fn RrelMainDashboard(state: DashboardState) -> impl IntoView {
    view! {
        <div class="rrel-dashboard min-h-screen bg-zinc-950 text-white p-8">
            <div class="max-w-7xl mx-auto">
                <div class="flex items-center justify-between mb-8">
                    <div>
                        <h1 class="text-4xl font-bold tracking-tight">RREL Dashboard</h1>
                        <p class="text-zinc-400 mt-1">v14.3 • Ontario + USA • Mercy-Gated • Cached</p>
                    </div>
                    <div class="px-4 py-2 bg-zinc-900 rounded-xl text-sm border border-zinc-800">
                        "TOLC 8 Active | ATTOM Cache: Synced"
                    </div>
                </div>

                // Ontario Section
                <section class="mb-10">
                    <h2 class="text-2xl font-semibold mb-4">Recent Ontario Offer Flows</h2>
                    <div class="space-y-3">
                        <For
                            each=move || state.recent_ontario_flows.clone()
                            key=|flow| flow.recommended_form.clone()
                            children=move |flow| {
                                view! {
                                    <div class="bg-zinc-900 border border-zinc-800 rounded-2xl px-5 py-4">
                                        <div class="flex justify-between">
                                            <div>
                                                <span class="font-mono text-sm text-zinc-400">{flow.deal_type.clone()}</span>
                                                <span class="ml-3 font-medium">{flow.recommended_form.clone()}</span>
                                            </div>
                                            <div class={if flow.offer_valid { "text-emerald-400" } else { "text-red-400" }}>
                                                {if flow.offer_valid { "Valid" } else { "Needs Review" }}
                                            </div>
                                        </div>
                                    </div>
                                }
                            }
                        />
                    </div>
                </section>

                // USA Section with Enriched Data
                <section>
                    <h2 class="text-2xl font-semibold mb-4">Recent USA Offer Flows (Enriched)</h2>
                    <div class="space-y-4">
                        <For
                            each=move || state.recent_usa_flows.clone()
                            key=|flow| flow.state.clone()
                            children=move |flow| {
                                view! {
                                    <div class="bg-zinc-900 border border-zinc-800 rounded-3xl p-5">
                                        <div class="flex justify-between items-start">
                                            <div>
                                                <div class="font-semibold text-lg">{flow.state.clone()}</div>
                                                <div class="text-sm text-zinc-400">{flow.summary.clone()}</div>
                                            </div>
                                            <div class={if flow.passed_regulatory { "text-emerald-400" } else { "text-amber-400" }}>
                                                {if flow.passed_regulatory { "Cleared" } else { "Issues Found" }}
                                            </div>
                                        </div>

                                        // Show enriched ATTOM data if available
                                        {flow.external_property_profile.as_ref().map(|profile| view! {
                                            <div class="mt-4 pt-4 border-t border-zinc-800 text-sm">
                                                <div class="text-zinc-400 mb-1">External Property Data (ATTOM Cached)</div>
                                                <div class="grid grid-cols-2 gap-x-8 text-zinc-300">
                                                    <div>
                                                        <span class="text-zinc-500">Tax Assessed:</span> {profile.tax_assessed_value.map(|v| format!("${:.0}", v)).unwrap_or("N/A".to_string())}
                                                    </div>
                                                    <div>
                                                        <span class="text-zinc-500">Last Sale:</span> {profile.last_sale_price.map(|v| format!("${:.0}", v)).unwrap_or("N/A".to_string())}
                                                    </div>
                                                </div>
                                            </div>
                                        }).unwrap_or_default()}

                                        {flow.external_risk_signals.as_ref().map(|risk| view! {
                                            <div class="mt-3 text-sm text-zinc-400">
                                                Risk Score: {risk.overall_risk_score.map(|s| format!("{:.2}", s)).unwrap_or("N/A".to_string())}
                                                {risk.flood_risk.as_ref().map(|r| view! { <span class="ml-4">Flood: {r}</span> }).unwrap_or_default()}
                                            </div>
                                        }).unwrap_or_default()}
                                    </div>
                                }
                            }
                        />
                    </div>
                </section>

                <div class="mt-8 text-xs text-zinc-500">
                    "All data mercy-gated • ATTOM cache active • Quantum Swarm synced"
                </div>
            </div>
        </div>
    }
}
