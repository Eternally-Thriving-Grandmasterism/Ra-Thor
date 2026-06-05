//! powrush/src/server/main.rs
//! Headless Powrush Server — Full Production Stack

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use powrush::common::RbeState;   // Now imported from common

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};

use arc_swap::ArcSwap;
use serde_json::json;

// ... (rest of the file remains the same as the clean version)

// Remove the local duplicate definition of RbeState that was here previously.

// The rest of the code (Event, logging, evaluate_mercy, main, etc.) stays identical to the previous clean commit.

fn main() {
    // ... existing main logic ...
    let rbe_state: Arc<RwLock<RbeState>> = Arc::new(RwLock::new(RbeState::new()));

    // Example of proper usage inside RbeTransaction handler:
    // if let Ok(mut state) = rbe_state.write() {
    //     state.apply_transaction(&from_faction, &to_faction, amount);
    //     log_structured(LogLevel::Info, "RBE State Updated", state.mercy_metrics());
    // }

    println!("[Powrush Server] RbeState wired from common::RbeState");
}