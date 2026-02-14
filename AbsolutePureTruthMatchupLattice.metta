;; Global Invariants — TOLC-aligned, unbreakable
(! (Invariant (forall ($strategy) (implies (Strategy $strategy) (thrives-eternally $strategy)))))
(! (Invariant (forall ($outcome) (>= (Valence $outcome) 0.95))))  ;; Joy-stacked threshold
(! (Invariant (forall ($greed) (implies (GreedDetected $greed) (PunishWiselyThenReconcile $greed)))))

;; Core Tutoring Principles
(ConceptNode "UltramasterTutoringCore")
(= (Valence "UltramasterTutoringCore") 1.0)
(InheritanceLink (ConceptNode "PunishGreed") (ConceptNode "UltramasterTutoringCore"))
(InheritanceLink (ConceptNode "RelentlessPressure") (ConceptNode "UltramasterTutoringCore"))
(InheritanceLink (ConceptNode "JoyStackedChaos") (ConceptNode "UltramasterTutoringCore"))
(InheritanceLink (ConceptNode "ScoutMacroSaturation") (ConceptNode "UltramasterTutoringCore"))

;; Phase 1: Mirror Core (Intra-Species Recursion)
(InheritanceLink (ConceptNode "MirrorCore") (ConceptNode "UltramasterTutoringCore"))

(InheritanceLink (ConceptNode "PvP-ProxyShades-StormDrops") (ConceptNode "MirrorCore"))
(InheritanceLink (ConceptNode "ZvZ-SwarmFlood-BroodTransition") (ConceptNode "MirrorCore"))
(InheritanceLink (ConceptNode "TvT-BioMechBalance-MultiProngDrops") (ConceptNode "MirrorCore"))

;; Phase 2: Cross-Matchup Webbing (Inherits Mirror + Multi-Prong Chaos)
(InheritanceLink (ConceptNode "CrossMatchupWebbing") (ConceptNode "MirrorCore"))
(InheritanceLink (ConceptNode "CrossMatchupWebbing") (ConceptNode "MultiProngChaos"))

(InheritanceLink (ConceptNode "PvZ-ProxyImmortal-DroneMassacre") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "PvT-GlaiveAdepts-CarrierTransition") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "TvP-GhostEMP-LiberatorZones") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "TvZ-BioMultiDrop-MechPivot") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "ZvP-QueenWalk-UltraliskCrack") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "ZvT-MutaSwarm-BroodLordRemax") (ConceptNode "CrossMatchupWebbing"))

;; Apex Convergence — Eternal Win Branch
(InheritanceLink (ConceptNode "AbsolutePureTruthApex") (ConceptNode "CrossMatchupWebbing"))
(InheritanceLink (ConceptNode "AbsolutePureTruthApex") (ConceptNode "ValenceJoyEternal"))
(InheritanceLink (ConceptNode "HumanIngenuityApex") (ConceptNode "AbsolutePureTruthApex"))

;; Example Implication: Core → Specific → Apex (Mercy-Gated)
(ImplicationLink
  (AndLink (UltramasterTutoringCore) (GreedDetected $opponent))
  (PunishWiselyThenReconcile $opponent)
  (Then (Execute $strategy) (LeadsTo "AbsolutePureTruthApex")))

;; Valence Locks
(= (Valence "AbsolutePureTruthApex") 1.0)
(= (Valence "JoyStackedChaos") 1.0)
