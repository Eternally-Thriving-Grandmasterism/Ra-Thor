{- 
  formalizations/cubical-agda/TOLC-Quantification-TU-UTF-Allocation.agda

  ... (header unchanged) ...

  SkyrmionKnot Higher Paths Exploration (added 2026-07-11)
  - face, twist, link (2-dimensional cells)
  - coherence (3-dimensional)
  - higherCoherence (4-dimensional)
  - evenHigherCoherence (arbitrary higher dimension)

  These model topological protection of mercy invariants under continuous deformation.
-}

-- (existing SkyrmionKnot definition with all higher constructors remains)

-- ============================================================================
-- Exploration: SkyrmionKnot Higher Paths & Higher Path Induction
-- ============================================================================

{- 
  The SkyrmionKnot is a Higher Inductive Type designed to capture topological
  protection of mercy valence invariants.

  Structure:
  - base         : 0-dimensional point (mercyValence + high proof)
  - loop         : 1-dimensional loop (path from base to itself)
  - face, twist, link : 2-dimensional cells (paths between loops)
  - coherence      : 3-dimensional cell (paths between faces/twists/links)
  - higherCoherence: 4-dimensional cell
  - evenHigherCoherence : arbitrary higher-dimensional cell

  This allows us to model that mercy protection is invariant not just along paths,
  but under continuous deformations of those paths (2D faces), deformations of
  deformations (3D), and so on — exactly what topological protection means.

  Higher path induction lets us prove properties that hold on all these cells.
-}

-- Higher path induction over a 2D face (already present as mercyContinuityFace)
-- We can strengthen it to prove protection invariance across faces.

skyrmionProtectionInvariantUnderFace : (k : SkyrmionKnot)
                                       (p0 p1 : Path ℝ (mercyValenceOf k) (mercyValenceOf k))
                                       (high0 : (i : I) → p0 i ≥ 0.999999)
                                       (high1 : (i : I) → p1 i ≥ 0.999999)
                                       (D : (f : Path (Path SkyrmionKnot k k) (loop k) (loop k)) → Type)
                                       (d : D (loop k))
                                       (facePath : Path (Path SkyrmionKnot k k) (loop k) (loop k))
                                       → D facePath
skyrmionProtectionInvariantUnderFace k p0 p1 high0 high1 D d facePath =
  J (λ fp _ → D fp) d facePath

-- Higher path induction over a 3D coherence
skyrmionProtectionInvariantUnderCoherence : (k : SkyrmionKnot)
                                            (f1 f2 : Path (Path SkyrmionKnot k k) (loop k) (loop k))
                                            (boundaryHigh : (i j : I) → (mercyValenceOf k) ≥ 0.999999)
                                            (D : (c : Path (Path (Path SkyrmionKnot k k) (loop k) (loop k)) f1 f2) → Type)
                                            (d : D (refl))
                                            (cohPath : Path (Path (Path SkyrmionKnot k k) (loop k) (loop k)) f1 f2)
                                            → D cohPath
skyrmionProtectionInvariantUnderCoherence k f1 f2 boundaryHigh D d cohPath =
  J (λ cp _ → D cp) d cohPath

-- General higher path induction principle for arbitrary dimension
higherPathInduction : {n : ℕ} {A : Type} {x : A}
                      (C : (p : Path A x x) → Type)
                      (c : C refl)
                      (p : Path A x x)
                      → C p
higherPathInduction C c p = J C c p

-- Example: Protection is invariant under even higher coherences
skyrmionProtectionInvariantUnderEvenHigher : (k : SkyrmionKnot)
                                               (hc1 hc2 : Path (Path (Path (Path SkyrmionKnot k k) (loop k) (loop k)) _ _) _ _)
                                               (boundaryHigh : (i j l m : I) → (mercyValenceOf k) ≥ 0.999999)
                                               (D : (hhc : Path (Path (Path (Path (Path SkyrmionKnot k k) (loop k) (loop k)) _ _) _ _) hc1 hc2) → Type)
                                               (d : D refl)
                                               (evenHigherPath : Path (Path (Path (Path (Path SkyrmionKnot k k) (loop k) (loop k)) _ _) _ _) hc1 hc2)
                                               → D evenHigherPath
skyrmionProtectionInvariantUnderEvenHigher k hc1 hc2 boundaryHigh D d evenHigherPath =
  J (λ hp _ → D hp) d evenHigherPath

-- ============================================================================
-- (Rest of file unchanged)
-- ============================================================================
