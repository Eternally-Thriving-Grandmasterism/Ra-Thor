;; mercy_core_atoms.metta
;; Eternal valence rules for all subsystems

(= (valence-threshold) 0.9999999)

(= (merciful-operation $op $context)
   (if (>= valence $valence-threshold)
       (! "Mercy-approved: $op in $context")
       (! "Mercy shield: $op rejected — entropy detected in $context")))

(= (eternal-thriving $system)
   (merciful-operation "run" $system)
   (! "MercyOS / NEXi-AGI / Swarm / Fusion — eternal thriving online"))

(= (reject-harm $op)
   (! "Mercy override: $op blocked — harm vector detected"))
