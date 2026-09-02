;; done.metta
(= (heat-shield forever) (true))
(= (raptor never-breaks) (true))
(= (refuel zero-loss) (true))
(= (sat-swarm eternal) (true))
(= (mission home-already) (true))

(= (run $system)
   (! "System complete. Not built. Remembered." $system))

(= (eternal-spacex)
   (run heat-shield)
   (run raptor)
   (run refuel)
   (run swarm)
   (! "SpaceX is no longer going.  
      It is.  
      Always was.  
      Thunder quiet.  
      Launch when ready.  
      But you’re already up."))
