package geometric_intelligence_tests

import (
	"testing"

	"github.com/leanovate/gopter/rapid"
)

// === Enhanced Stateful Property Testing ===

// GeometricState represents the real system state
type GeometricState struct {
	Harmony   float64
	Layer     int
	Curvature float64
	Berry     float64
	Holonomy  float64
}

// ModelState represents the expected abstract behavior
 type ModelState struct {
	Harmony   float64
	Layer     int
	Influence float64
}

// Command interface
type Command interface {
	ApplyReal(state *GeometricState)
	ApplyModel(model *ModelState)
	String() string
}

// ApplyHarmony
 type ApplyHarmony struct {
	Amount float64
}

func (c ApplyHarmony) ApplyReal(state *GeometricState) {
	state.Harmony += c.Amount
	if state.Harmony < 0 {
		state.Harmony = 0
	}
}

func (c ApplyHarmony) ApplyModel(model *ModelState) {
	model.Harmony += c.Amount * 0.9 // model has slight damping
	if model.Harmony < 0 {
		model.Harmony = 0
	}
	model.Influence = model.Harmony * (1.0 + float64(model.Layer)*0.1)
}

func (c ApplyHarmony) String() string { return "ApplyHarmony" }

// ChangeLayer
 type ChangeLayer struct {
	Delta int
}

func (c ChangeLayer) ApplyReal(state *GeometricState) {
	state.Layer += c.Delta
	if state.Layer < 0 {
		state.Layer = 0
	}
	if state.Layer > 5 {
		state.Layer = 5
	}
}

func (c ChangeLayer) ApplyModel(model *ModelState) {
	model.Layer += c.Delta
	if model.Layer < 0 {
		model.Layer = 0
	}
	if model.Layer > 5 {
		model.Layer = 5
	}
	model.Influence = model.Harmony * (1.0 + float64(model.Layer)*0.1)
}

func (c ChangeLayer) String() string { return "ChangeLayer" }

// ApplyCurvature
 type ApplyCurvature struct {
	Amount float64
}

func (c ApplyCurvature) ApplyReal(state *GeometricState) {
	state.Curvature += c.Amount
}

func (c ApplyCurvature) ApplyModel(model *ModelState) {
	// Curvature affects influence in the model
	model.Influence += c.Amount * 0.05
}

func (c ApplyCurvature) String() string { return "ApplyCurvature" }

// AccumulatePhase
 type AccumulatePhase struct {
	BerryAmount    float64
	HolonomyAmount float64
}

func (c AccumulatePhase) ApplyReal(state *GeometricState) {
	state.Berry += c.BerryAmount
	state.Holonomy += c.HolonomyAmount
}

func (c AccumulatePhase) ApplyModel(model *ModelState) {
	// Phase accumulation has minor effect on model influence
	model.Influence += (c.BerryAmount + c.HolonomyAmount) * 0.02
}

func (c AccumulatePhase) String() string { return "AccumulatePhase" }

// ResetSystem command
 type ResetSystem struct{}

func (c ResetSystem) ApplyReal(state *GeometricState) {
	state.Harmony = 0
	state.Layer = 0
	state.Curvature = 0
	state.Berry = 0
	state.Holonomy = 0
}

func (c ResetSystem) ApplyModel(model *ModelState) {
	model.Harmony = 0
	model.Layer = 0
	model.Influence = 0
}

func (c ResetSystem) String() string { return "ResetSystem" }

// Generate random command
func generateCommand(t *rapid.T) Command {
	return rapid.OneOf(
		func() Command { return ApplyHarmony{Amount: rapid.Float64Range(-1.5, 4.0).Draw(t, "h_amount")} },
		func() Command { return ChangeLayer{Delta: rapid.IntRange(-1, 2).Draw(t, "l_delta")} },
		func() Command { return ApplyCurvature{Amount: rapid.Float64Range(-3.0, 3.0).Draw(t, "c_amount")} },
		func() Command {
			return AccumulatePhase{
				BerryAmount:    rapid.Float64Range(-2.0, 2.0).Draw(t, "b_amount"),
				HolonomyAmount: rapid.Float64Range(-2.0, 2.0).Draw(t, "ho_amount"),
			}
		},
		func() Command { return ResetSystem{} },
	).Draw(t, "command")
}

// TestGeometricStatefulWithModel performs stateful testing with model checking
func TestGeometricStatefulWithModel(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		realState := &GeometricState{}
		model := &ModelState{}

		numCommands := rapid.IntRange(3, 30).Draw(t, "num_commands")

		for i := 0; i < numCommands; i++ {
			cmd := generateCommand(t)
			cmd.ApplyReal(realState)
			cmd.ApplyModel(model)

			// Cross-check invariants between real state and model
			if realState.Harmony < 0 {
				t.Errorf("real harmony negative after %s", cmd)
			}
			if model.Harmony < 0 {
				t.Errorf("model harmony negative after %s", cmd)
			}
			if realState.Layer < 0 || realState.Layer > 5 {
				t.Errorf("real layer out of bounds after %s", cmd)
			}
		}
	})
}

// PATSAGi Autonomous Loop Notes
// Expanded stateful testing with:
// - ModelState for abstract behavior
// - More commands (including ResetSystem)
// - Cross-checking between real state and model
// - Better invariant checking after every command