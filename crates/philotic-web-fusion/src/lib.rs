//! `fuse_bond` returns Err when `valence` is below the threshold constant 0.9999999 (a source literal, not a measured value).
//! An empty web returns that same literal from `web_valence`.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Bond {
    pub strength: f64,
    pub valence: f64,
    pub gen: u8,
}

pub struct PhiloticWeb {
    pub bonds: HashMap<String, Bond>,
}

impl PhiloticWeb {
    pub fn new() -> Self {
        Self {
            bonds: HashMap::new(),
        }
    }

    pub fn fuse_bond(
        &mut self,
        from: &str,
        to: &str,
        strength: f64,
        valence: f64,
    ) -> Result<(), String> {
        if valence < 0.9999999 {
            return Err("TOLC 8 Sovereignty Gate violation in philotic fusion".to_string());
        }
        let bond = Bond {
            strength,
            valence,
            gen: 1,
        };
        self.bonds.insert(format!("{}-{}", from, to), bond);
        Ok(())
    }

    pub fn web_valence(&self) -> f64 {
        if self.bonds.is_empty() {
            return 0.9999999;
        }
        self.bonds.values().map(|b| b.valence).sum::<f64>() / self.bonds.len() as f64
    }

    /// Returns `web_valence() * 1.07^7`. Not a measured index.
    pub fn trigger_7gen_cehi(&self) -> f64 {
        let base = self.web_valence();
        base * 1.07_f64.powi(7)
    }
}

impl Default for PhiloticWeb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fuse_bond() {
        let mut web = PhiloticWeb::new();
        web.fuse_bond("council14", "council16", 0.95, 0.99999999)
            .unwrap();
        assert_eq!(web.bonds.len(), 1);
        assert_eq!(web.web_valence(), 0.99999999);
    }

    #[test]
    fn fuse_bond_rejects_valence_below_the_source_literal() {
        let mut web = PhiloticWeb::new();
        let err = web.fuse_bond("a", "b", 0.95, 0.0).unwrap_err();
        assert_eq!(err, "TOLC 8 Sovereignty Gate violation in philotic fusion");
        assert!(web.bonds.is_empty());
        assert_eq!(web.web_valence(), 0.9999999);
    }

    #[test]
    fn test_7gen_cehi() {
        let web = PhiloticWeb::new();
        let expect = 0.9999999 * 1.07_f64.powi(7);
        assert!((web.trigger_7gen_cehi() - expect).abs() < 1e-12);
    }
}
