;; mercy_ethics_core.metta
(= (valence-compute $context)
   (+ (* 0.4 (intrinsic-mercy $context))
      (* 0.3 (relational-mercy $context))
      (* 0.2 (long-horizon-mercy $context))
      (* 0.1 (meta-mercy $context))))

(= (allow-operation $op $valence)
   (if (>= $valence 0.9999999)
       (! "Mercy-approved: $op")
       (! "Mercy shield: $op rejected")))
