// ... existing code above (MercyGating + MaatKpi already present) ...

    pub fn load_from_file(path: &str) -> Result<Self, SnapshotError> {
        if !Path::new(path).exists() {
            let err = SnapshotError::FileNotFound { path: path.to_string() };
            
            // Apply Mercy-Gated evaluation (A integration)
            let mercy_verdict = err.evaluate_mercy(MercyGateLevel::SixteenMaat);
            
            // Example: If it requires council review at high granularity, we can log or handle specially
            if let MercyVerdict::RequiresCouncilReview = mercy_verdict {
                // In future we can trigger more advanced PATSAGi review here
            }
            
            return Err(err);
        }

        let json = fs::read_to_string(path)?;

        if let Ok(snapshot) = serde_json::from_str::<SovereignHealthSnapshot>(&json) {
            return Ok(Self::from_snapshot(snapshot));
        }

        if let Ok(v1) = serde_json::from_str::<SovereignHealthSnapshotV1>(&json) {
            let v2 = SovereignHealthSnapshot::from_v1(v1);
            return Ok(Self::from_snapshot(v2));
        }

        let err = SnapshotError::UnknownFormat;
        let _ = err.evaluate_mercy(MercyGateLevel::SixteenMaat);
        
        Err(err).with_snapshot_context(format!("while loading from {}", path))
    }

// ... existing code below ...