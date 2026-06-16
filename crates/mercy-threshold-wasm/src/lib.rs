use wasmtime::{Engine, Instance, Module, Store, TypedFunc};
use anyhow::Result;

/// WASM bridge for the formal Lean Mercy Threshold Theorem.
/// This path always uses the Lean4 exported `check_mercy_threshold`.
pub struct MercyThresholdBridge {
    store: Store<()>,
    check_fn: TypedFunc<(u32, u32, bool, f64), i32>,
}

impl MercyThresholdBridge {
    pub fn new(wasm_bytes: &[u8]) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;

        let check_fn = instance.get_typed_func::<(u32, u32, bool, f64), i32>(
            &mut store,
            "check_mercy_threshold",
        )?;

        Ok(Self { store, check_fn })
    }

    pub fn check(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<bool> {
        let result = self.check_fn.call(&mut self.store, (vertices, faces, chiral, mercy_valence))?;
        Ok(result != 0)
    }
}

/// Hybrid native path (only available with `native` feature).
/// Directly uses the production Rust implementation from the mial crate
/// (evaluate_geometry_mercy_component + full MWPO logic).
#[cfg(feature = "native")]
pub mod native {
    use mial::mwpo::{MercyWeightedPreferenceOptimization, GeometryParams, MercyContext, MercyGeometryScore};
    use anyhow::Result;

    /// High-level hybrid checker.
    /// In native mode this calls the full mial implementation directly
    /// (bypassing WASM for speed + richer scoring).
    pub fn evaluate_geometry_mercy_hybrid(
        geometry: &GeometryParams,
        mercy_context: &MercyContext,
        council_id: u32,
    ) -> Result<MercyGeometryScore> {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        mwpo.evaluate_geometry_mercy_component(geometry, mercy_context, council_id)
            .map_err(|e| anyhow::anyhow!("MIAL geometry mercy evaluation failed: {}", e))
    }

    /// Re-export for convenience
    pub use mial::mwpo::{GeometryParams, MercyContext, MercyGeometryScore, MercyWeightedPreferenceOptimization};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_bridge_smoke() -> Result<()> {
        let _engine = Engine::default();
        Ok(())
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_hybrid_path() {
        use native::*;
        // Basic smoke that the mial path is callable
        let geo = GeometryParams {
            solid_type: mial::mwpo::SacredSolid::Platonic,
            dimensions: 3,
            symmetry_group: mial::mwpo::SymmetryGroup { order: 48, chiral: false },
            evolution_step: 10,
            particle_density: 0.6,
            lattice_config: None,
        };
        let ctx = MercyContext {
            active_gates: vec![],
            valence: 0.999999,
            council_id: 7,
        };
        // This would actually run the full evaluation if gates allow
        let _ = evaluate_geometry_mercy_hybrid(&geo, &ctx, 7);
    }
}