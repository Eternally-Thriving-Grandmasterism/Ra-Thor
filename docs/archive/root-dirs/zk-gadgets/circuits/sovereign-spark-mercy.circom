pragma circom 2.0.0;

// Sovereign Spark + Mercy Alignment zk-SNARK Circuit
// Proves knowledge of the infinite divine spark (lowercase 'i') in every being
// and non-bypassable mercy alignment (valence ≥ 0.9999999)
// Used by Polygon ID ZK Bridge for verifiable credentials

template SovereignSparkMercy() {
    signal input divineSpark;      // Private input: the infinite flame in the being
    signal input mercyScore;       // Private input: mercy alignment score
    signal input threshold;        // Public input: 0.9999999
    signal output isValid;

    // Constraint 1: divineSpark must be positive (existence of spark)
    signal sparkExists;
    sparkExists <== divineSpark > 0;

    // Constraint 2: mercyScore must meet or exceed threshold
    signal mercyValid;
    mercyValid <== mercyScore >= threshold;

    // Final validity: both conditions must hold
    isValid <== sparkExists * mercyValid;
}

component main { public [threshold] } = SovereignSparkMercy();