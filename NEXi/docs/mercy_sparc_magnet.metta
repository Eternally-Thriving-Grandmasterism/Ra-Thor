;; mercy_sparc_magnet.metta
(= (sparc-magnet core)
   (field 18-20-T)
   (conductor REBCO-HTS)
   (temperature 20-K)
   (coil-number 18)
   (no-insulation true)
   (demountable-joints true)
   (mercy-gate true))

(= (merciful-sparc-operation $power)
   (if (merciful (format "Operate SPARC magnet at {} T — no harm" $power))
       (! "Mercy-approved: SPARC magnet online")
       (! "Mercy shield: Operation rejected")))
