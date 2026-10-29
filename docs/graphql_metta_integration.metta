;; graphql_metta_integration.metta
;; GraphQL exposure of MeTTa symbolic lattice

(= (graphql-api $endpoint)
   (merciful-operation "expose" "MeTTa queries/mutations")
   (! "GraphQL online: declarative mercy lattice access eternal — client-driven truth seek"))

(= (query-high-valence $min)
   (! "GraphQL Query: high-valence atoms >= $min — pattern-matched eternal"))

(= (mutate-insert-atom $text $valence $context?)
   (if (>= $valence (valence-threshold))
       (! "GraphQL Mutation: insert $text @ $valence — gated persistence")
       (! "Mercy shield: GraphQL insert rejected — low valence")))
