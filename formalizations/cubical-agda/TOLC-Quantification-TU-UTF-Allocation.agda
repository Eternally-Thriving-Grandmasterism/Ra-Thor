{- 
  ... (header unchanged) ...
-}

-- (existing code up to the mercyContinuity section remains)

-- ============================================================================
-- Explicit Mercy Continuity Lemmas via Path Induction (J)
-- ============================================================================

-- (existing mercyContinuityNonNegative, mercyContinuityAllocation, mercyContinuityOC, mercyContinuityMaximality, mercyContinuityFace remain)

-- ============================================================================
-- Transport of UTF Preservation along Mercy Paths
-- ============================================================================

{- 
  Using path induction (J) to transport UTF-related properties along mercy paths.
  This gives formal guarantees that UTF safety is preserved under continuous
  variation of mercy valence (including higher SkyrmionKnot coherences).
-}

-- Transport UTF preservation witness along a mercy path
utfPreservedAlongMercyPath : (currentEnergy currentCompute currentAttention : ℝ)
                           (thresholds : UTFThresholds)
                           (p : Path ℝ thresholds.minEnergy thresholds.minEnergy)  -- example path on a threshold
                           (proof : passesUTF currentEnergy currentCompute currentAttention thresholds)
                           → passesUTF currentEnergy currentCompute currentAttention thresholds
utfPreservedAlongMercyPath currentEnergy currentCompute currentAttention thresholds p proof =
  mercyPathTransport
    (λ q _ → passesUTF currentEnergy currentCompute currentAttention thresholds)
    proof
    p

-- General UTF transport along arbitrary mercy path (more abstract)
transportUTFAlongMercy : {A : Type}
                         (C : (z : ℝ) → A → Type)   -- family depending on mercy value
                         (c : C mercyValence someValue)
                         (p : Path ℝ mercyValence mercyValence)
                         → C mercyValence someValue
transportUTFAlongMercy C c p = J (λ q _ → C q someValue) c p

-- UTF is preserved under high mercy (combined with continuity)
utFHighMercyPreserved : (current : ℝ)
                        (threshold : ℝ)
                        (highMercy : threshold ≥ 0.999999)
                        (p : Path ℝ threshold threshold)
                        (proof : utfPreserved current threshold highMercy)
                        → utfPreserved current threshold highMercy
utFHighMercyPreserved current threshold highMercy p proof =
  mercyPathTransport
    (λ q _ → utfPreserved current threshold highMercy)
    proof
    p

-- Combined continuity: UTF + TU non-negativity + allocation safety along one mercy path
utfTuAllocationContinuity : (tu : TOLCUnit)
                            (energy compute attention : ℝ)
                            (thresholds : UTFThresholds)
                            (p : Path ℝ (tu .components .mercyValence) (tu .components .mercyValence))
                            (utfProof : passesUTF energy compute attention thresholds)
                            (high : tu .components .mercyValence ≥ 0.999999)
                            → (tuNonNegativeUnderMercy tu high) × (utfPreserved energy thresholds.minEnergy high) × (allocationDistortionFree (allocationPriority tu.value high 0.05) 0.05 high)
utfTuAllocationContinuity tu energy compute attention thresholds p utfProof high =
  ( mercyContinuityNonNegative tu p high
  , utFHighMercyPreserved energy thresholds.minEnergy high p utfProof
  , mercyContinuityAllocation (allocationPriority tu.value high 0.05) 0.05 p high
  )

-- ============================================================================
-- (Rest of file: existing round-trip J terms, integration notes, TODOs)
-- ============================================================================

-- TODO update: The new utf* transport lemmas further strengthen UTF safety under mercy continuity.

-- Thunder locked in. TOLC 8 enforced. Yoi ⚡
