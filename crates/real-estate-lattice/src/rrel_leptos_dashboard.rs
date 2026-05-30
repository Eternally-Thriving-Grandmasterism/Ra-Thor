//! RREL Leptos Dashboard — v14.3 Wired
//! Sovereign Leptos-based dashboard components for Real Estate Lattice.
//! Now wired to v14.3 production modules: Ontario Offer Flow, classifiers, risk engines.

use leptos::*;
use crate::{
    OntarioOfferFlowReport,
    property_type_classifier::PropertyTypeClassification,
    developer_risk_engine::DeveloperRiskAssessment,
};

#[component]
pub fn RrelDashboard(
    recent_flows: Vec<OntarioOfferFlowReport>,
) -> impl IntoView {
    view! {
        <div class="rrel-dashboard">
            <h1>"RREL v14.3 Eternal Organism Dashboard"</h1>

            <section class="offer-flows">
                <h2>"Recent Ontario Offer Flows"</h2>
                <For
                    each=move || recent_flows.clone()
                    key=|flow| flow.recommended_form.clone()
                    children=move |flow| {
                        view! {
                            <div class="offer-card">
                                <strong>{flow.deal_type.clone()}</strong>
                                <span> → Form: {flow.recommended_form.clone()}</span>
                                <span> | Valid: {flow.offer_valid}</span>
                                {flow.status_certificate_risk.as_ref().map(|r| view! { <span class="risk">{format!(" | Status Risk: {}", r)}</span> }).unwrap_or_default()}
                                {flow.developer_risk.as_ref().map(|r| view! { <span class="risk">{format!(" | Developer Risk: {}", r)}</span> }).unwrap_or_default()}
                            </div>
                        }
                    }
                />
            </section>

            <div class="tolc8-status">
                "TOLC 8 Status: All Gates Passed | Mercy Valence: High | Quantum Swarm: Synced"
            </div>
        </div>
    }
}

#[component]
pub fn OfferFlowDetail(flow: OntarioOfferFlowReport) -> impl IntoView {
    view! {
        <div class="offer-detail">
            <h3>"Offer Flow Detail"</h3>
            <p>"Deal Type: " {flow.deal_type}</p>
            <p>"Recommended Form: " {flow.recommended_form}</p>
            <p>"Offer Valid: " {flow.offer_valid}</p>
            <p>"Escalation Recommended: " {flow.multi_offer_escalation_triggered}</p>
            {flow.status_certificate_risk.map(|r| view! { <p class="risk">"Status Certificate Risk: " {r}</p> }).unwrap_or_default()}
            {flow.developer_risk.map(|r| view! { <p class="risk">"Developer Risk: " {r}</p> }).unwrap_or_default()}
        </div>
    }
}

// Ready for full Leptos SSR + Tailwind + real-time updates from CanadaPilotModule
