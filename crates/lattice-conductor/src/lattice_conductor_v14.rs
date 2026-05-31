    pub fn new() -> Self {
        let gates = vec![
            MercyGate::Truth, MercyGate::Order, MercyGate::Love, MercyGate::Compassion,
            MercyGate::Service, MercyGate::Abundance, MercyGate::Joy, MercyGate::CosmicHarmony,
        ];
        let mut rules = std::collections::HashMap::new();
        rules.insert("Ontario".to_string(), "RESA/TRESA compliance + reverse onus safety checks".to_string());
        rules.insert("USA".to_string(), "State-level disclosure + federal fair housing".to_string());

        // Custom shard count for better parallelism under high load
        // 32 shards provides good balance for typical offer processing workloads
        let attom_cache: DashMap<String, Arc<AttomData>> =
            DashMap::with_shard_amount(32);

        LatticeConductor {
            version: "v14.4.0-geometric-intelligence",
            mercy_gates: gates,
            attom_cache,
            regulatory_rules: rules,
            geometric_engine: PolyhedralHarmonicEngine::new(),
        }
    }