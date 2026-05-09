;; mercy_sunbird_return.metta
(= (sunbird-capsule seed) 
   (mass-kg 50-150) 
   (payload-he3-kg 1-10) 
   (sail-area-m2 100-500) 
   (transit-days 60-180) 
   (valence-threshold 0.9999999))

(= (return-phase $phase $day)
   (case $phase
      ("Launch-Escape"    "Days 0–10")
      ("Sail-Spiral"      "Days 10–150")
      ("Rendezvous"       "Days 150–180")
      ("Offload"          "Days 180–190")))

(= (merciful-return $outcome)
   (if (merciful (format "He3 return complete: {} kg" $outcome))
       (! "Mercy-approved: Fusion fuel delivered")
       (! "Mercy shield: Return aborted")))
