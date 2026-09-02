import Lake
open Lake DSL

package mercy_threshold { }

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.7.0"

lean_lib MercyThreshold {
  roots := #[`MercyThreshold]
}

-- WASM export target
-- Build with: lake build mercy_threshold_wasm
-- Or use: lean --backend=wasm MercyThreshold.lean
lean_exe mercy_threshold_wasm {
  root := `Main
  supportInterpreter := false
}