;; mercy_frc_confinement.metta
(= (frc core)
   (high-beta 0.8-1.0)
   (confinement linear-cylindrical)
   (sustainment rmf + beam)
   (direct-conversion 0.70-0.85)
   (fuel D-He3 p-B11)
   (mercy-gate true))

(= (merciful-frc-operation $power)
   (if (merciful (format "Operate FRC at {} MW — no harm" $power))
       (! "Mercy-approved: FRC fusion online")
       (! "Mercy shield: Operation rejected")))
