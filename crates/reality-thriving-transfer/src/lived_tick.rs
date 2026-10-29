//! T1 — optional lived-tick JSON reader (R&D only).
//!
//! Env: `POWRUSH_LIVED_TICK_PATH`. Unset ⇒ no-op (`Ok(None)`).
//! No network. Never flips `POWRUSH_INGEST`. Not RTT bridging.

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Env var that enables the optional lived-tick file read.
pub const POWRUSH_LIVED_TICK_PATH_ENV: &str = "POWRUSH_LIVED_TICK_PATH";

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LivedClimate {
    #[serde(default)]
    pub harmony: Option<f64>,
    #[serde(default)]
    pub stress: Option<f64>,
    #[serde(default)]
    pub tons: Option<f64>,
    #[serde(default)]
    pub restored: Option<f64>,
    #[serde(default)]
    pub hex_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LivedStanding {
    #[serde(default)]
    pub peace: Option<bool>,
    #[serde(default)]
    pub declared_lethal: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LivedWeek {
    #[serde(default)]
    pub tons: Option<f64>,
    #[serde(default)]
    pub restored: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LivedHourFlags {
    #[serde(default)]
    pub satchel: Option<bool>,
    #[serde(default)]
    pub flow: Option<bool>,
    #[serde(default)]
    pub reserve: Option<bool>,
    #[serde(default)]
    pub hour_two: Option<bool>,
    #[serde(default)]
    pub hour_three: Option<bool>,
}

/// Soft parse of `powrush_lived_tick_v1` — fields optional so partial yard exports still summarize.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PowrushLivedTick {
    pub schema: String,
    #[serde(default)]
    pub house_id: Option<String>,
    #[serde(default)]
    pub house_name: Option<String>,
    #[serde(default)]
    pub climate: Option<LivedClimate>,
    #[serde(default)]
    pub standing: Option<LivedStanding>,
    #[serde(default)]
    pub week: Option<LivedWeek>,
    #[serde(default)]
    pub hour_flags: Option<LivedHourFlags>,
}

pub fn parse_powrush_lived_tick_json(json: &str) -> Result<PowrushLivedTick, String> {
    let tick: PowrushLivedTick = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid powrush_lived_tick_v1 JSON: {}", e))?;
    if tick.schema != "powrush_lived_tick_v1" {
        return Err(format!(
            "Mercy Gate (Truth): expected schema powrush_lived_tick_v1, got '{}'",
            tick.schema
        ));
    }
    Ok(tick)
}

/// One-line mercy summary for R&D logs — never invents `n_online`.
pub fn mercy_summary_line(tick: &PowrushLivedTick) -> String {
    let house = tick
        .house_name
        .as_deref()
        .or(tick.house_id.as_deref())
        .unwrap_or("unknown-house");
    let (harmony, stress) = tick
        .climate
        .as_ref()
        .map(|c| (c.harmony.unwrap_or(0.0), c.stress.unwrap_or(0.0)))
        .unwrap_or((0.0, 0.0));
    let peace = tick
        .standing
        .as_ref()
        .and_then(|s| s.peace)
        .unwrap_or(false);
    let lethal = tick
        .standing
        .as_ref()
        .and_then(|s| s.declared_lethal)
        .unwrap_or(false);
    let posture = if lethal {
        "lethal-declared"
    } else if peace {
        "peace-held"
    } else {
        "peace-unset"
    };
    format!(
        "lived-tick mercy | house={} | harmony={:.3} stress={:.3} | {} | keys=never",
        house, harmony, stress, posture
    )
}

pub fn load_lived_tick_from_path(path: &Path) -> Result<(PowrushLivedTick, String), String> {
    let raw = fs::read_to_string(path).map_err(|e| {
        format!(
            "Mercy Gate (Truth): cannot read lived tick at {}: {}",
            path.display(),
            e
        )
    })?;
    let tick = parse_powrush_lived_tick_json(&raw)?;
    let line = mercy_summary_line(&tick);
    Ok((tick, line))
}

/// If `POWRUSH_LIVED_TICK_PATH` is unset, returns `Ok(None)` (no-op).
pub fn load_lived_tick_from_env() -> Result<Option<(PowrushLivedTick, String)>, String> {
    match std::env::var_os(POWRUSH_LIVED_TICK_PATH_ENV) {
        None => Ok(None),
        Some(os) if os.is_empty() => Ok(None),
        Some(os) => {
            let path = Path::new(&os);
            let pair = load_lived_tick_from_path(path)?;
            Ok(Some(pair))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/lived_tick_sample.json");

    #[test]
    fn parse_sample_and_summary() {
        let tick = parse_powrush_lived_tick_json(FIXTURE).unwrap();
        assert_eq!(tick.house_name.as_deref(), Some("Heartwood"));
        let line = mercy_summary_line(&tick);
        assert!(line.contains("Heartwood"));
        assert!(line.contains("peace-held"));
        assert!(line.contains("keys=never"));
        assert!(!line.to_lowercase().contains("n_online"));
    }

    #[test]
    fn reject_wrong_schema() {
        let bad = r#"{"schema":"nope","house_name":"X"}"#;
        assert!(parse_powrush_lived_tick_json(bad).is_err());
    }

    #[test]
    fn env_unset_is_noop() {
        // Ensure we do not accidentally inherit a path in CI.
        std::env::remove_var(POWRUSH_LIVED_TICK_PATH_ENV);
        let out = load_lived_tick_from_env().unwrap();
        assert!(out.is_none());
    }
}
