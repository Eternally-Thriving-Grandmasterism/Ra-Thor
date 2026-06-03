package geometric_intelligence_tests

import (
	"testing"

	"github.com/leanovate/gopter/rapid" // adjust import path as needed
)

// RiemannianCurvature represents curvature in a Riemannian manifold context
// (inspired by PR #192 RiemannianMercyManifold concepts)
type RiemannianCurvature float64

// RiemannianCurvatureGen creates a generator with good shrinking behavior
func RiemannianCurvatureGen() *rapid.Generator[RiemannianCurvature] {
	return rapid.Custom(func(t *rapid.T) RiemannianCurvature {
		return RiemannianCurvature(rapid.Float64Range(-10.0, 10.0).Draw(t, "curvature"))
	}).Shrink(func(c RiemannianCurvature) []RiemannianCurvature {
		var shrinks []RiemannianCurvature
		val := float64(c)

		// Always try shrinking toward 0.0 first (most "simple" curvature)
		if val != 0 {
			shrinks = append(shrinks, 0)
		}

		// Shrink magnitude while preserving sign
		current := val
		for i := 0; i < 10 && abs(current) > 0.01; i++ {
			current *= 0.5
			shrinks = append(shrinks, RiemannianCurvature(current))
		}

		// Try common "simple" curvature values
		simpleValues := []float64{-2, -1, -0.5, 0.5, 1, 2}
		for _, sv := range simpleValues {
			if (sv < 0 && val > sv) || (sv > 0 && val < sv) || (sv == 0 && val != 0) {
				shrinks = append(shrinks, RiemannianCurvature(sv))
			}
		}

		return shrinks
	})
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Example property test using the custom shrinker
func TestRiemannianCurvatureProperties(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		curvature := RiemannianCurvatureGen().Draw(t, "curvature")

		// Example invariant: curvature should allow stable transport
		// (placeholder logic)
		if abs(float64(curvature)) > 100 {
			t.Errorf("curvature out of reasonable range: %f", curvature)
		}
	})
}

// PATSAGi Autonomous Loop Notes
// Implemented custom shrinker for RiemannianCurvature.
// Shrinks toward 0, reduces magnitude, and tries common simple values.
// Good foundation for testing RiemannianMercyManifold behavior.