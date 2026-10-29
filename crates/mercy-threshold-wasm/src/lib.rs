//! Loads caller-supplied wasm bytes when feature `native` is enabled.
//! `MercyThresholdBridge::new` compiles those bytes with wasmtime and looks up export names.
//! This crate does not embed a proof.
//! Score methods return the export, or the source literal `0.0` when that export is missing.
//! `0.0` is not a measured score.
//! `native::evaluate_geometry_mercy_hybrid` calls `mial` MWPO. Those scores are source-literal formulas.
//! Contact: info@Rathor.ai

#![cfg(feature = "native")]

use anyhow::Result;
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

/// Host-side wasmtime bridge. The third argument of each export is `i32` (`0` or `1`).
/// Rust `bool` is not a wasm value type in wasmtime 25.
pub struct MercyThresholdBridge {
    store: Store<()>,
    check_fn: TypedFunc<(u32, u32, i32, f64), i32>,
    all_gates_fn: Option<TypedFunc<(u32, u32, i32, f64), i32>>,
    status_fn: Option<TypedFunc<(u32, u32, i32, f64), u32>>,
    love_fn: Option<TypedFunc<(u32, u32, i32), f32>>,
    truth_fn: Option<TypedFunc<(u32, u32, i32), f32>>,
    mercy_fn: Option<TypedFunc<(u32, u32, i32, f64), f32>>,
    abundance_fn: Option<TypedFunc<(u32, u32, i32), f32>>,
    joy_fn: Option<TypedFunc<(u32, u32, i32, f64), f32>>,
    harmony_fn: Option<TypedFunc<(u32, u32, i32), f32>>,
    resonance_fn: Option<TypedFunc<(u32, u32, i32), f32>>,
}

impl MercyThresholdBridge {
    pub fn new(wasm_bytes: &[u8]) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;

        let check_fn = instance
            .get_typed_func::<(u32, u32, i32, f64), i32>(&mut store, "check_mercy_threshold")?;

        let all_gates_fn = instance
            .get_typed_func::<(u32, u32, i32, f64), i32>(&mut store, "check_all_gates_strong")
            .ok();

        let status_fn = instance
            .get_typed_func::<(u32, u32, i32, f64), u32>(&mut store, "get_mercy_threshold_status")
            .ok();

        let love_fn = instance
            .get_typed_func::<(u32, u32, i32), f32>(&mut store, "get_love_score")
            .ok();
        let truth_fn = instance
            .get_typed_func::<(u32, u32, i32), f32>(&mut store, "get_truth_score")
            .ok();
        let mercy_fn = instance
            .get_typed_func::<(u32, u32, i32, f64), f32>(&mut store, "get_mercy_score")
            .ok();
        let abundance_fn = instance
            .get_typed_func::<(u32, u32, i32), f32>(&mut store, "get_abundance_score")
            .ok();
        let joy_fn = instance
            .get_typed_func::<(u32, u32, i32, f64), f32>(&mut store, "get_joy_score")
            .ok();
        let harmony_fn = instance
            .get_typed_func::<(u32, u32, i32), f32>(&mut store, "get_harmony_score")
            .ok();
        let resonance_fn = instance
            .get_typed_func::<(u32, u32, i32), f32>(&mut store, "get_geometry_resonance")
            .ok();

        Ok(Self {
            store,
            check_fn,
            all_gates_fn,
            status_fn,
            love_fn,
            truth_fn,
            mercy_fn,
            abundance_fn,
            joy_fn,
            harmony_fn,
            resonance_fn,
        })
    }

    pub fn check(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<bool> {
        Ok(self.check_fn.call(
            &mut self.store,
            (vertices, faces, i32::from(chiral), mercy_valence),
        )? != 0)
    }

    pub fn check_all_gates_strong(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<bool> {
        match &mut self.all_gates_fn {
            Some(fn_ptr) => Ok(fn_ptr.call(
                &mut self.store,
                (vertices, faces, i32::from(chiral), mercy_valence),
            )? != 0),
            None => self.check(vertices, faces, chiral, mercy_valence),
        }
    }

    /// Returns the `get_love_score` export, or the source literal `0.0` when it is missing.
    pub fn get_love_score(&mut self, vertices: u32, faces: u32, chiral: bool) -> Result<f32> {
        self.love_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(&mut self.store, (vertices, faces, i32::from(chiral)))
        })
    }

    /// Returns the `get_truth_score` export, or the source literal `0.0` when it is missing.
    pub fn get_truth_score(&mut self, vertices: u32, faces: u32, chiral: bool) -> Result<f32> {
        self.truth_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(&mut self.store, (vertices, faces, i32::from(chiral)))
        })
    }

    /// Returns the `get_mercy_score` export, or the source literal `0.0` when it is missing.
    pub fn get_mercy_score(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<f32> {
        self.mercy_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(
                &mut self.store,
                (vertices, faces, i32::from(chiral), mercy_valence),
            )
        })
    }

    /// Returns the `get_abundance_score` export, or the source literal `0.0` when it is missing.
    pub fn get_abundance_score(&mut self, vertices: u32, faces: u32, chiral: bool) -> Result<f32> {
        self.abundance_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(&mut self.store, (vertices, faces, i32::from(chiral)))
        })
    }

    /// Returns the `get_joy_score` export, or the source literal `0.0` when it is missing.
    pub fn get_joy_score(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<f32> {
        self.joy_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(
                &mut self.store,
                (vertices, faces, i32::from(chiral), mercy_valence),
            )
        })
    }

    /// Returns the `get_harmony_score` export, or the source literal `0.0` when it is missing.
    pub fn get_harmony_score(&mut self, vertices: u32, faces: u32, chiral: bool) -> Result<f32> {
        self.harmony_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(&mut self.store, (vertices, faces, i32::from(chiral)))
        })
    }

    /// Returns the `get_geometry_resonance` export, or the source literal `0.0` when it is missing.
    pub fn get_geometry_resonance(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
    ) -> Result<f32> {
        self.resonance_fn.as_mut().map_or(Ok(0.0), |f| {
            f.call(&mut self.store, (vertices, faces, i32::from(chiral)))
        })
    }

    pub fn get_proof_status(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<u32> {
        match &mut self.status_fn {
            Some(fn_ptr) => fn_ptr.call(
                &mut self.store,
                (vertices, faces, i32::from(chiral), mercy_valence),
            ),
            None => Ok(if self.check(vertices, faces, chiral, mercy_valence)? {
                1
            } else {
                0
            }),
        }
    }

    /// Lists whether each optional export was present. It does not report a measured score.
    pub fn summary(&self) -> String {
        format!(
            "MercyThresholdBridge v14.15.0 | all_gates={} | status={} | love={} truth={} mercy={} abundance={} joy={} harmony={} resonance={}",
            self.all_gates_fn.is_some(),
            self.status_fn.is_some(),
            self.love_fn.is_some(),
            self.truth_fn.is_some(),
            self.mercy_fn.is_some(),
            self.abundance_fn.is_some(),
            self.joy_fn.is_some(),
            self.harmony_fn.is_some(),
            self.resonance_fn.is_some()
        )
    }
}

#[cfg(feature = "native")]
pub mod native {
    use anyhow::Result;
    pub use mial::mwpo::{
        GeometryParams, MercyContext, MercyGeometryScore, MercyWeightedPreferenceOptimization,
    };

    pub fn evaluate_geometry_mercy_hybrid(
        geometry: &GeometryParams,
        mercy_context: &MercyContext,
        council_id: u32,
    ) -> Result<MercyGeometryScore> {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        mwpo.evaluate_geometry_mercy_component(geometry, mercy_context, council_id)
            .map_err(|e| anyhow::anyhow!("MIAL geometry mercy evaluation failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mial::mwpo::{GeometryParams, MercyContext, SacredSolid, SymmetryGroup};

    #[test]
    fn empty_wasm_bytes_are_rejected() {
        assert!(MercyThresholdBridge::new(&[]).is_err());
    }

    #[test]
    fn native_platonic_zero_resonance_is_the_source_literal_0_96() {
        let geometry = GeometryParams {
            solid_type: SacredSolid::Platonic,
            dimensions: 3,
            symmetry_group: SymmetryGroup {
                order: 0,
                chiral: false,
            },
            evolution_step: 0,
            particle_density: 0.0,
            lattice_config: None,
        };
        let context = MercyContext {
            active_gates: vec![],
            valence: 0.999999,
            council_id: 4,
        };
        let score = native::evaluate_geometry_mercy_hybrid(&geometry, &context, 4)
            .expect("overall is above the source literal 0.92");
        assert_eq!(score.geometry_resonance, 0.96);
        assert_eq!(score.harmony, 0.96);
        assert_eq!(score.truth, 0.999);
        assert_eq!(score.love, (0.96_f64 * 0.97 + 0.999999 * 0.03).min(1.0));
        assert_eq!(score.mercy, (0.96_f64 * 0.92 + 0.08).min(1.0));
        assert_eq!(score.abundance, (0.96_f64 * 0.88 + 0.12).min(1.0));
    }

    #[test]
    fn native_valence_below_the_source_literal_is_an_error() {
        let geometry = GeometryParams {
            solid_type: SacredSolid::Platonic,
            dimensions: 3,
            symmetry_group: SymmetryGroup {
                order: 0,
                chiral: true,
            },
            evolution_step: 0,
            particle_density: 0.0,
            lattice_config: None,
        };
        let context = MercyContext {
            active_gates: vec![],
            valence: 0.999998,
            council_id: 4,
        };
        assert!(native::evaluate_geometry_mercy_hybrid(&geometry, &context, 4).is_err());
    }
}
