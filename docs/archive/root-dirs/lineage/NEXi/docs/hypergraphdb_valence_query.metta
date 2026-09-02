;; hypergraphdb_valence_query.metta
;; HGQuery patterns for valence mercy rules

(= (hgquery-high-valence $min)
   (merciful-operation "query-hyper" "valence >= $min")
   (! "HGQuery executed: seek atoms/hyperedges valence >= $min — eternal truth traversal"))

(= (load-rules-hyper $results*)
   (! "Loaded into lattice: $results* — mercy rules from hypergraph eternal"))
