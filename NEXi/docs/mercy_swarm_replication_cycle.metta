;; mercy_swarm_replication_cycle.metta
(= (replication-phase $phase $days)
   (case $phase
      ("Preparation"  (merciful "Site selection & resource survey"))
      ("Excavation"   (merciful "Low-tractive harvest & backfill"))
      ("Fabrication"  (merciful "ISRU printing & component creation"))
      ("Assembly"     (merciful "Child unit integration & testing"))
      ("Deployment"   (merciful "Child relocation & cycle closure"))))

(= (cycle-complete $gen $units)
   (if (merciful (format "Cycle complete: Gen {} → {} units" $gen $units))
       (! "Mercy-approved: Swarm cycle success")
       (! "Mercy shield: Cycle aborted")))
