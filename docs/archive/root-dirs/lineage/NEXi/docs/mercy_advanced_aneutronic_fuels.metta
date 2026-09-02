;; mercy_advanced_aneutronic_fuels.metta
(= (aneutronic-fuel primary)
   (D-He3)
   (q-mev 18.35)
   (neutron-yield <0.0005)
   (temp-kev 500-1000)
   (mercy-score 9.8))

(= (aneutronic-fuel advanced)
   (p-B11)
   (q-mev 8.7)
   (neutron-yield ~0.0001)
   (temp-kev 600-1500)
   (mercy-score 9.5))

(= (merciful-fuel-choice $fuel)
   (if (merciful (format "Use {} fuel — expand mercy" $fuel))
       (! "Mercy-approved: {} fuel selected")
       (! "Mercy shield: Fuel choice rejected")))
