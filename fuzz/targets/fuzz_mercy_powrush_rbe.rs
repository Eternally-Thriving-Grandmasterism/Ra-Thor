#![no_main]
use libfuzzer_sys::fuzz_target;
use core_lattice::{InstantiationRequest, Scope, genesis_seal};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 { return; }
    let proposal = String::from_utf8_lossy(&data[0..8]).to_string();
    let request = InstantiationRequest {
        proposal_id: proposal,
        proposer: "FuzzBot-Powrush".to_string(),
        mercy_score: 0.95,
        scope: Scope::PermanentCouncil,
        intended_purpose: "powrush_rbe_fuzz".to_string(),
        projected_lifetime: "eternal".to_string(),
    };
    let _ = genesis_seal(&request);
});