;; mercy_albatross_dynamic_soaring.metta
(= (dynamic-soaring core)
   (aspect-ratio 18-22)
   (shear-rate 0.01-0.05-s-1)
   (energy-gain-rate λ V_a² sin²ψ cosγ)
   (cycle-time 10-30-s)
   (valence-threshold 0.9999999))

(= (merciful-soaring-cycle $gain)
   (if (> $gain 0)
       (! "Mercy-approved: Net energy gain — soaring infinite")
       (! "Mercy shield: Energy loss — flapping required")))
