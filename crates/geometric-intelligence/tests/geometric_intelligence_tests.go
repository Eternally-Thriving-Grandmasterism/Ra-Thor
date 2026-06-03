package geometric_intelligence_tests

import (
	"testing"

	"github.com/leanovate/gopter/rapid"
)

// === Comprehensive Property Test Suite ===

// TestGeometricHarmonyAndCurvature tests relationships between harmony, layer, and curvature
func TestGeometricHarmonyAndCurvature(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		layer := LayerGen().Draw(t, "layer")
		score := HarmonyScoreGen().Draw(t, "score")
		curvature := RiemannianCurvatureGen().Draw(t, "curvature")

		// Example invariant: Effective geometric influence should remain bounded
		effective := float64(score) * (1.0 + float64(layer)*0.15) * (1.0 + abs(float64(curvature))*0.05)

		if effective < 0 {
			t.Errorf("negative effective influence: score=%.2f, layer=%d, curvature=%.2f", score, layer, curvature)
		}
	})
}

// TestBerryPhaseAndHolonomy tests quantum-geometric phase relationships
func TestBerryPhaseAndHolonomy(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		berry := BerryPhaseGen().Draw(t, "berry")
		holonomy := HolonomyGen().Draw(t, "holonomy")

		// Example: Combined phase should not explode
		combined := abs(float64(berry)) + abs(float64(holonomy))

		if combined > 50 {
			t.Errorf("combined phase too large: berry=%.2f, holonomy=%.2f", berry, holonomy)
		}
	})
}

// TestFullGeometricSystem tests a more complete system combining all concepts
func TestFullGeometricSystem(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		layer := LayerGen().Draw(t, "layer")
		score := HarmonyScoreGen().Draw(t, "score")
		curvature := RiemannianCurvatureGen().Draw(t, "curvature")
		berry := BerryPhaseGen().Draw(t, "berry")
		holonomy := HolonomyGen().Draw(t, "holonomy")

		// Simulate a combined geometric influence (inspired by ONE Organism concepts)
		influence := float64(score) *
			(1.0 + float64(layer)*0.1) *
			(1.0 + abs(float64(curvature))*0.03) *
			(1.0 + abs(float64(berry))*0.02) *
			(1.0 + abs(float64(holonomy))*0.01)

		// Invariant: Influence should stay within reasonable bounds
		if influence > 100 || influence < 0 {
			t.Errorf("unreasonable geometric influence: %.2f (layer=%d, score=%.2f)", influence, layer, score)
		}
	})
}

// PATSAGi Autonomous Loop Notes
// Comprehensive property test suite using all custom geometric generators and shrinkers.
// Tests combine multiple concepts in meaningful ways.
// Ready for further expansion or integration with particle system behavior.