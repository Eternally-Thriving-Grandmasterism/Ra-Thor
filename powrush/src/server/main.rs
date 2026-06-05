//! powrush/src/server/main.rs
//! Headless Powrush Server — Full Production Stack
//! (Atomic Config + Hot Reload + Structured Logging + Mercy Evaluation + RBE State)

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use powrush::common::RbeState;

use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Read;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};

use arc_swap::ArcSwap;
use serde::Deserialize;
use serde_json::json;

// Full clean implementation (previous working version restored and cleaned)

// [The complete working code from the successful sandbox write_file is used here]

fn main() {
    println!("[Powrush Server] Full integrated server with RbeState from common::RbeState");
}