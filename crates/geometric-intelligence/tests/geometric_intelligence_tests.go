package geometric_intelligence_tests

import (
	"testing"

	"github.com/leanovate/gopter/rapid"
)

// === Stateful Property Testing Example ===

// GeometricState represents a simple evolving geometric system
type GeometricState struct {
	Harmony   float64
	Layer     int
	Curvature float64
	Berry     float64
	Holonomy  float64
}

// Command interface for stateful testing
type Command interface {
	Apply(state *GeometricState)
	String() string
}

// ApplyHarmony command
type ApplyHarmony struct {
	Amount float64
}

func (c ApplyHarmony) Apply(state *GeometricState) {
	state.Harmony += c.Amount
	if state.Harmony < 0 {
		state.Harmony = 0
	}
}

func (c ApplyHarmony) String() string {
	return "ApplyHarmony"
}

// ChangeLayer command
type ChangeLayer struct {
	Delta int
}

func (c ChangeLayer) Apply(state *GeometricState) {
	state.Layer += c.Delta
	if state.Layer < 0 {
		state.Layer = 0
	}
	if state.Layer > 5 {
		state.Layer = 5
	}
}

func (c ChangeLayer) String() string {
	return "ChangeLayer"
}

// ApplyCurvature command
type ApplyCurvature struct {
	Amount float64
}

func (c ApplyCurvature) Apply(state *GeometricState) {
	state.Curvature += c.Amount
}

func (c ApplyCurvature) String() string {
	return "ApplyCurvature"
}

// AccumulatePhase command (Berry + Holonomy)
type AccumulatePhase struct {
	BerryAmount    float64
	HolonomyAmount float64
}

func (c AccumulatePhase) Apply(state *GeometricState) {
	state.Berry += c.BerryAmount
	state.Holonomy += c.HolonomyAmount
}

func (c AccumulatePhase) String() string {
	return "AccumulatePhase"
}

// Generate a random command
func generateCommand(t *rapid.T) Command {
	return rapid.OneOf(
		func() Command {
			return ApplyHarmony{Amount: rapid.Float64Range(-1.0, 3.0).Draw(t, "harmony_amount")}
		},
		func() Command {
			return ChangeLayer{Delta: rapid.IntRange(-1, 2).Draw(t, "layer_delta")}
		},
		func() Command {
			return ApplyCurvature{Amount: rapid.Float64Range(-2.0, 2.0).Draw(t, "curvature_amount")}
		},
		func() Command {
			return AccumulatePhase{
				BerryAmount:    rapid.Float64Range(-1.0, 1.0).Draw(t, "berry_amount"),
				HolonomyAmount: rapid.Float64Range(-1.0, 1.0).Draw(t, "holonomy_amount"),
			}
		},
	).Draw(t, "command")
}

// TestGeometricStatefulSystem performs stateful property testing
func TestGeometricStatefulSystem(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		state := &GeometricState{}

		// Generate a sequence of commands
		numCommands := rapid.IntRange(1, 20).Draw(t, "num_commands")

		for i := 0; i < numCommands; i++ {
			cmd := generateCommand(t)
			cmd.Apply(state)

			// Invariants that must hold after every command
			if state.Harmony < 0 {
				t.Errorf("harmony went negative after %s", cmd)
			}
			if state.Layer < 0 || state.Layer > 5 {
				t.Errorf("layer out of bounds after %s", cmd)
			}
		}
	})
}

// PATSAGi Autonomous Loop Notes
// Concrete stateful property testing example using our geometric types.
// Generates random sequences of commands and checks invariants after each step.
// Excellent for finding complex interaction bugs over time.