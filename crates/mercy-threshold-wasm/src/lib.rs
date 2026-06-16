use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

pub struct MercyThresholdBridge {
    store: Store<()>,
    check_fn: TypedFunc<(u32, u32, bool, f64), i32>,
}

impl MercyThresholdBridge {
    pub fn new(wasm_bytes: &[u8]) -> anyhow::Result<Self> {
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

    /// Returns `true` if the input passes the Mercy Threshold Theorem
    pub fn check(
        &mut self,
        vertices: u32,
        faces: u32,
        chiral: bool,
        mercy_valence: f64,
    ) -> anyhow::Result<bool> {
        let result = self.check_fn.call(
            &mut self.store,
            (vertices, faces, chiral, mercy_valence),
        )?;
        Ok(result != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j27_sovereignty() {
        // This would load the actual .wasm file in a real test
        // For now, this is a placeholder structure
        // TODO: Load build/mercy_threshold.wasm and assert on known Johnson solid J27
    }
}