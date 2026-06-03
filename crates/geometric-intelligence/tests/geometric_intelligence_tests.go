package geometric_intelligence_tests

import (
	"sort"
	"testing"

	"github.com/leanovate/gopter/rapid"
)

// RiemannianCurvature represents curvature in a Riemannian manifold context
type RiemannianCurvature float64

// RiemannianCurvatureGen with refined shrinking toward zero
func RiemannianCurvatureGen() *rapid.Generator[RiemannianCurvature] {
	return rapid.Custom(func(t *rapid.T) RiemannianCurvature {
		return RiemannianCurvature(rapid.Float64Range(-10.0, 10.0).Draw(t, "curvature"))
	}).Shrink(func(c RiemannianCurvature) []RiemannianCurvature {
		val := float64(c)
		if val == 0 {
			return nil // already at simplest value
		}

		var shrinks []RiemannianCurvature

		// 1. Always try 0 first (strongest preference for zero)
		shrinks = append(shrinks, 0)

		// 2. Systematically shrink toward zero with smaller steps
		sign := 1.0
		if val < 0 {
			sign = -1
		}

		current := val
		for i := 0; i < 12 && abs(current) > 0.001; i++ {
			// Move closer to zero more aggressively than simple halving
			step := current * 0.6
			current -= step
			shrinks = append(shrinks, RiemannianCurvature(current))
		}

		// 3. Try common simple curvature values, sorted by closeness to zero
		simple := []float64{-2, -1, -0.5, 0.5, 1, 2}
		sort.Slice(simple, func(i, j int) bool {
			return abs(simple[i]) < abs(simple[j])
		})
		for _, s := range simple {
			if (sign > 0 && s > 0 && s < val) || (sign < 0 && s < 0 && s > val) {
				shrinks = append(shrinks, RiemannianCurvature(s))
			}
		}

		// Remove duplicates and sort by closeness to zero
		seen := make(map[float64]bool)
		var unique []RiemannianCurvature
		for _, s := range shrinks {
			if !seen[float64(s)] {
				seen[float64(s)] = true
				unique = append(unique, s)
			}
		}

		// Sort so values closest to zero come first (better shrinking order)
		sort.Slice(unique, func(i, j int) bool {
			return abs(float64(unique[i])) < abs(float64(unique[j]))
		})

		return unique
	})
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Example usage in property test
func TestRiemannianCurvatureRefined(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		curv := RiemannianCurvatureGen().Draw(t, "curvature")
		// ... test invariants ...
		_ = curv
	})
}

// PATSAGi Autonomous Loop Notes (Cycle 11)
// Refined shrinking toward zero:
// - Always tries 0 first
// - Uses more aggressive steps toward zero
// - Sorts candidates by closeness to zero
// - Removes duplicates
// Better minimal failing cases for Riemannian curvature properties.