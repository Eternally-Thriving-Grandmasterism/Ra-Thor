;; mercy_cfs_tokamak.metta
(= (cfs-tokamak core)
   (high-field 18-20-T)
   (hts-magnets true)
   (fuel D-T)
   (q-target >2-10)
   (direct-conversion false)
   (mercy-gate true))

(= (merciful-cfs-operation $power)
   (if (merciful (format "Operate CFS tokamak at {} MW — no harm" $power))
       (! "Mercy-approved: CFS fusion online")
       (! "Mercy shield: Operation rejected")))
