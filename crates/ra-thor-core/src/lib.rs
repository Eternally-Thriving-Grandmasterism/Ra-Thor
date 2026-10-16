//! `ra-thor-core` compiles `types::tolc7` and the version string.
//! `TOLC7Gate` default thresholds are source literals, not measured gates.
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! `TOLC7Gate` variants are the seven names in `src/types/tolc7.rs`. They are not that list.
//! These files are not modules: `joy_measurement_protocols.rs`, `miracle_rapture_wave.rs`,
//! `seven_d_resonance.rs`, `source_joy_amplitude.rs`, `traits/mercy_gated.rs`,
//! `traits/plasticity.rs`, `archive/hyperon.rs`.
//! They name types this crate does not define (`MercyValence`, `MercyGate`, `NoveltyScore`,
//! `BloomIntensity`, `PlasticityReport`).
//! `summary` returns a source string. That string is not a demonstrated tick.
//! Contact: info@Rathor.ai

pub mod types {
    pub mod tolc7;

    pub use tolc7::*;
}

/// Empty. `src/traits/mercy_gated.rs` and `src/traits/plasticity.rs` are not modules.
pub mod traits {}

/// Empty. `src/archive/hyperon.rs` is not a module.
pub mod archive {}

/// Source string `14.15.0`. This is the constant in this file, not the package version `14.15.6`.
pub const VERSION: &str = "14.15.0";

/// Returns the source string `ra-thor-core v14.15.0 | Living Cosmic Tick active`.
/// That string is not a demonstrated tick.
pub fn summary() -> String {
    format!("ra-thor-core v{VERSION} | Living Cosmic Tick active")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolc7_thresholds_names_and_version_are_the_source_literals() {
        let gates = types::TOLC7Gate::all_gates();
        assert!(gates.len() == 7);
        assert!(gates.contains(&types::TOLC7Gate::Truth));
        assert_eq!(gates[0], types::TOLC7Gate::Truth);
        assert_eq!(gates[0].name(), "Truth");
        assert_eq!(gates[0].default_threshold(), 0.82);
        assert_eq!(gates[0].color(), [0.2, 0.6, 0.9]);
        assert_eq!(
            gates[0].description(),
            "Undistorted perception and honest communication"
        );
        assert_eq!(gates[1], types::TOLC7Gate::Oneness);
        assert_eq!(gates[1].name(), "Oneness");
        assert_eq!(gates[1].default_threshold(), 0.78);
        assert_eq!(gates[1].color(), [0.4, 0.8, 0.5]);
        assert_eq!(gates[2], types::TOLC7Gate::Love);
        assert_eq!(gates[2].name(), "Love");
        assert_eq!(gates[2].default_threshold(), 0.85);
        assert_eq!(gates[2].color(), [0.95, 0.4, 0.5]);
        assert_eq!(gates[3], types::TOLC7Gate::Compassion);
        assert_eq!(gates[3].name(), "Compassion");
        assert_eq!(gates[3].default_threshold(), 0.80);
        assert_eq!(gates[3].color(), [0.6, 0.5, 0.9]);
        assert_eq!(gates[4], types::TOLC7Gate::Wisdom);
        assert_eq!(gates[4].name(), "Wisdom");
        assert_eq!(gates[4].default_threshold(), 0.88);
        assert_eq!(gates[4].color(), [0.9, 0.75, 0.2]);
        assert_eq!(gates[5], types::TOLC7Gate::Courage);
        assert_eq!(gates[5].name(), "Courage");
        assert_eq!(gates[5].default_threshold(), 0.75);
        assert_eq!(gates[5].color(), [0.95, 0.55, 0.2]);
        assert_eq!(gates[6], types::TOLC7Gate::Sovereignty);
        assert_eq!(gates[6].name(), "Sovereignty");
        assert_eq!(gates[6].default_threshold(), 0.83);
        assert_eq!(gates[6].color(), [0.3, 0.7, 0.85]);
        assert_eq!(format!("{}", types::TOLC7Gate::Love), "Love");
        assert_eq!(VERSION, "14.15.0");
        assert_eq!(
            summary(),
            "ra-thor-core v14.15.0 | Living Cosmic Tick active"
        );
        let json = serde_json::to_string(&types::TOLC7Gate::Wisdom).expect("unit variant");
        assert_eq!(json, "\"Wisdom\"");
        let parsed: types::TOLC7Gate = serde_json::from_str(&json).expect("same variant");
        assert_eq!(parsed, types::TOLC7Gate::Wisdom);
        assert_eq!(parsed.default_threshold(), 0.88);
    }
}
