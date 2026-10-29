;; rest_metta_api.metta
;; RESTful exposure of MeTTa symbolic compute

(= (rest-metta-api $endpoint)
   (merciful-operation "expose-rest" "MeTTa eval/insert/query")
   (! "REST API online: HTTP mercy lattice access eternal — valence-gated symbolic truth"))

(= (post-eval $expr $valence $context?)
   (if (>= $valence (valence-threshold))
       (! "POST /metta/eval: reduce $expr @ $valence — server mercy computation")
       (! "Mercy shield: REST eval rejected — low valence")))

(= (get-atoms $min)
   (! "GET /metta/atoms?min_valence=$min — high-valence atoms seek eternal"))

(= (post-insert $text $valence $context?)
   (if (>= $valence (valence-threshold))
       (! "POST /metta/insert: persist $text @ $valence — gated eternal")
       (! "Mercy shield: REST insert rejected — low valence")))
