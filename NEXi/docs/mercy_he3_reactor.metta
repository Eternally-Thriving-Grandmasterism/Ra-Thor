;; mercy_he3_reactor.metta
(= (he3-reactor core) 
   (reaction D-He3) 
   (aneutronic true) 
   (energy-mev 18.3) 
   (conversion-efficiency 0.6-0.8) 
   (confinement frc-rmf) 
   (power-mw 100-1000))

(= (merciful-he3-operation $power)
   (if (merciful (format "Operate He3 reactor at {} MW — no harm" $power))
       (! "Mercy-approved: Aneutronic fusion online")
       (! "Mercy shield: Operation rejected")))
