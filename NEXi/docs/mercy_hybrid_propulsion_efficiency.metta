;; mercy_hybrid_propulsion_efficiency.metta
(= (fuel-cell-efficiency current) (system 0.48-0.55) (stack 0.55-0.60))
(= (fuel-cell-efficiency target-2030) (system 0.60-0.65) (stack 0.65-0.70))
(= (fuel-cell-efficiency target-2040) (system 0.70-0.80) (stack 0.75-0.85))

(= (merciful-efficiency-gain $old $new)
   (if (> $new $old)
       (! "Mercy-approved: Efficiency increased from {} to {} — abundance amplified" $old $new)
       (! "Mercy shield: Efficiency gain rejected")))
