use wasmtime::{Engine, Instance, Module, Store, TypedFunc};
use anyhow::Result;

/// WASM bridge for the formal Lean Mercy Threshold Theorem.
///
/// Key formal guarantees (see lean/mercy-threshold/MercyThreshold.lean):
/// - `check_mercy_threshold_true_implies_all_gates_strong` (bridge lemma)
/// - `mercy_threshold_safety_implies_all_gates_strong` (master lemma)
///
/// New programmatic export: `check_all_gates_strong` returns true when all gates are proven strong.
pub struct MercyThresholdBridge {
    store: Store<()>,
    check_fn: TypedFunc<(u32, u32, bool, f64), i32>,
    all_gates_fn: Option<TypedFunc<(u32, u32, bool, f64), i32>>,
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

        let all_gates_fn = instance.get_typed_func::<(u32, u32, bool, f64), i32>(
            &mut store,
            "check_all_gates_strong",
        ).ok();

        let status_fn = instance.get_typed_func::<(u32, u32, bool, f64), u32>(
            &mut store,
            "get_mercy_threshold_status",
        ).ok();

        Ok(Self { store, check_fn, all_gates_fn, status_fn })
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

    /// Returns true if all gates (Love, Mercy, Truth, Abundance, Harmony, Joy, geometry_resonance)
    /// are proven strong by the Lean formalization.
    pub fn check_all_gates_strong(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> Result<bool> {
        match &mut self.all_gates_fn {
            Some(fn_ptr) => {
                let result = fn_ptr.call(&mut self.store, (vertices, faces, chiral, mercy_valence))?;
                Ok(result != 0)
            }
            None => self.check(vertices, faces, chiral, mercy_valence),
        }
    }

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