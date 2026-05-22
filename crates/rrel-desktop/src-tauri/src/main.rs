//! RREL Desktop — Production Tauri commands with real form processing
use real_estate_lattice::*;
use tauri::Manager;

#[tauri::command]
fn create_brokerage_offer(base: OfferPackage, fees: FeeStructure) -> Result<AssembledBrokerageOffer, String> {
    RrelBrokerageAssembler::new().assemble_brokerage_offer(base, fees)
}

#[tauri::command]
fn process_offer_to_purchase(offer: OfferToPurchase) -> Result<RecoForm, String> {
    RecoFormHandlers::new().process_offer_to_purchase(offer, None)
}

#[tauri::command]
fn process_buyer_representation(agreement: BuyerRepresentationAgreement) -> Result<RecoForm, String> {
    RecoFormHandlers::new().process_buyer_representation(agreement, None)
}

#[tauri::command]
fn get_rrel_status() -> String {
    "RREL v3.2 Eternal Organism • TOLC 8 Sealed • Mercy-Gated • Lattice Conductor + Quantum Swarm Active • One Organism".to_string()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            create_brokerage_offer,
            process_offer_to_purchase,
            process_buyer_representation,
            get_rrel_status
        ])
        .run(tauri::generate_context!())
        .expect("RREL Desktop failed to run");
}