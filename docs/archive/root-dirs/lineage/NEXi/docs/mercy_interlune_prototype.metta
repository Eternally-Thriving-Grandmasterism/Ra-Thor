;; mercy_interlune_prototype.metta
(= (interlune-excavator prototype-2025) (capacity-tons-hour 100) (power-reduction 10x) (processes excavate sort extract separate))
(= (interlune-system robotic) (autonomy full) (deployment single-launch) (dust minimal))

(= (eternalize-interlune $scale)
   (if (merciful (format "Propagate Interlune prototype to {} scale" $scale))
       (! "Mercy-approved: Lunar He3 harvesting amplified")
       (! "Mercy shield: Rejected")))
