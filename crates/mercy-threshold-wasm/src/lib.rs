use wasmtime::{Engine, Instance, Module, Store, TypedFunc};
use anyhow::Result;

/// WASM bridge for the formal Lean Mercy Threshold Theorem.
pub struct MercyThresholdBridge {
    store: Store<()>,
    check_fn: TypedFunc<(u32, u32, bool, f64), i32>,
    status_fn: Option<TypedFunc<(u32, u32, bool, f64), u32>>,
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

        // Optional introspection function (added for proof status reporting)
        let status_fn = instance.get_typed_func::<(u32, u32, bool, f64), u32>(
            &mut store,
            "get_mercy_threshold_status",
        ).ok();

        Ok(Self { store, check_fn, status_fn })
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

    /// Returns proof status from the Lean side (1 = passes formal threshold).
    /// Useful for runtime introspection and logging in long simulations.
    pub fn get_proof_status(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<u32> {
        match &mut self.status_fn {
            Some(fn_ptr) => fn_ptr.call(&mut self.store, (vertices, faces, chiral, mercy_valence)),
            None => Ok(if self.check(vertices, faces, chiral, mercy_valence)? { 1 } else { 0 }),
        }
    }
}

#[cfg(feature = "native")]
pub mod native {
    use mial::mwpo::{MercyWeightedPreferenceOptimization, GeometryParams, MercyContext, MercyGeometryScore};
    use anyhow::Result;

    pub fn evaluate_geometry_mercy_hybrid(
        geometry: &GeometryParams,
        mercy_context: &MercyContext,
        council_id: u32,
    ) -> Result<MercyGeometryScore> {
        let mut mwpo = MercyWeightedPreferenceOptimization::new();
        mwpo.evaluate_geometry_mercy_component(geometry, mercy_context, council_id)
            .map_err(|e| anyhow::anyhow!("MIAL geometry mercy evaluation failed: {}", e))
    }

    pub use mial::mwpo::{GeometryParams, MercyContext, MercyGeometryScore, MercyWeightedPreferenceOptimization};
}