#![no_main]
use libfuzzer_sys::fuzz_target;
use core_lattice::{InstantiationRequest, Scope, genesis_seal};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 { return; }
    let proposal = String::from_utf8_lossy(&data[0..8]).to_string();
    let request = InstantiationRequest {
        proposal_id: proposal,
        proposer: "FuzzBot-RestAPI".to_string(),
        mercy_score: 0.7,
        scope: Scope::ExploratoryBranch,
        intended_purpose: "rest_api_fuzz".to_string(),
        projected_lifetime: "medium".to_string(),
    };
    let _ = genesis_seal(&request);
});