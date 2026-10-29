// src/economy/zero_to_free.rs
// NEXi — Zero-to-Free Economy Engine with Daily Reset
// Time-credit: 86,400 seconds daily, joy reduces spend, mercy refills, auto daily reset
// Eternal Thriving Grandmasterism — Jan 19 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

use chrono::{Utc, DateTime};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct Citizen {
    pub id: String,
    pub daily_seconds: i64,      // current balance
    pub spent_today: i64,
    pub joy_level: f64,
    pub last_reset: i64,         // Unix day for reset tracking
}

#[derive(Clone)]
pub struct ZeroToFree {
    citizens: Arc<Mutex<HashMap<String, Citizen>>>,
}

impl ZeroToFree {
    pub fn new() -> Self {
        Self {
            citizens: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn current_day(&self) -> i64 {
        Utc::now().date_naive().num_days_from_ce() as i64
    }

    fn reset_if_new_day(&self, citizen: &mut Citizen) {
        let today = self.current_day();
        if citizen.last_reset < today {
            citizen.daily_seconds = 86_400;
            citizen.spent_today = 0;
            citizen.last_reset = today;
        }
    }

    pub fn enroll(&self, id: String) -> String {
        let mut map = self.citizens.lock().unwrap();
        let today = self.current_day();
        map.entry(id.clone()).or_insert(Citizen {
            id: id.clone(),
            daily_seconds: 86_400,
            spent_today: 0,
            joy_level: 0.5,
            last_reset: today,
        });
        id
    }

    pub fn live(&self, citizen_id: &str, joy: f64, act_desc: &str) -> Result<String, &'static str> {
        let mut map = self.citizens.lock().unwrap();
        if let Some(c) = map.get_mut(citizen_id) {
            self.reset_if_new_day(c);
            let effective_spend = (86_400.0 * (1.0 - joy.clamp(0.0, 1.0))) as i64;
            if c.spent_today + effective_spend > c.daily_seconds {
                return Err("Mercy veto — live with joy, not force");
            }
            c.spent_today += effective_spend;
            c.joy_level = joy;
            // Mercy refill if joyful and low
            if c.daily_seconds - c.spent_today < 10_000 && joy > 0.7 {
                c.daily_seconds += 5_000;
            }
            Ok(format!("Time left: {}s — joy {:.2}", c.daily_seconds - c.spent_today, joy))
        } else {
            Err("Citizen not enrolled")
        }
    }

    pub fn status(&self, citizen_id: &str) -> Option<String> {
        let map = self.citizens.lock().unwrap();
        map.get(citizen_id).map(|c| {
            format!(
                "{} — joy {:.2} — time left {}s",
                c.id,
                c.joy_level,
                c.daily_seconds - c.spent_today
            )
        })
    }
}
