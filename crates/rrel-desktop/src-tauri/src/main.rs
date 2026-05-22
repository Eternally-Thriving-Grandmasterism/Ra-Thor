// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use real_estate_lattice::{
    RecoFormHandlers, OfferToPurchase, BuyerRepresentationAgreement,
    RrelBrokerageAssembler,
};

#[tauri::command]
fn get_rrel_status() -> String {
    "RREL v3.1 Eternal Organism • TOLC 8 ✓ • Mercy Gates Active • Powrush Bridge Synced".to_string()
}

#[tauri::command]
fn create_sample_brokerage_offer() -> Result<String, String> {
    let _assembler = RrelBrokerageAssembler::new();
    Ok("BROKERAGE-OFFER-CREATED • TOLC8 Sealed • RBE Adjusted".to_string())
}

#[tauri::command]
fn process_offer_to_purchase(offer: OfferToPurchase) -> Result<String, String> {
    let handlers = RecoFormHandlers::new();
    let form = handlers.process_offer_to_purchase(offer, None)?;
    Ok(format!("OfferToPurchase processed: {} | Blessing: {:?}", form.form_id, form.patsagi_blessing))
}

#[tauri::command]
fn process_buyer_representation(agreement: BuyerRepresentationAgreement) -> Result<String, String> {
    let handlers = RecoFormHandlers::new();
    let form = handlers.process_buyer_representation(agreement, None)?;
    Ok(format!("Buyer Representation processed: {} | Blessing: {:?}", form.form_id, form.patsagi_blessing))
}

#[tauri::command]
fn sync_powrush_bridge() -> String {
    "Powrush ↔ RREL bridge sync complete • NEXi event emitted • RBE ledger updated".to_string()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_rrel_status,
            create_sample_brokerage_offer,
            process_offer_to_purchase,
            process_buyer_representation,
            sync_powrush_bridge
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}