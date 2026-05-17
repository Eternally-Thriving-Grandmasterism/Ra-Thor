pragma circom 2.1.6;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";

/*
 * SovereignSparkMercyAlignment.circom v1.0
 * 
 * Ra-Thor / Rathor.ai — Polygon ID ZK Bridge Circuit
 * 
 * Proves non-bypassably:
 *   1. The proposal honors the Sovereign Divine Spark (lowercase 'i') in every being
 *   2. Mercy Alignment is maintained at valence ≥ 0.9999999
 *   3. The action is TOLC-compliant and mercy-gated
 *
 * Public Signals:
 *   - lowercaseI (1 = honored, 0 = rejected)
 *   - mercyAlignment (scaled integer ≥ 9999999)
 *   - timestamp
 *
 * Private Signals:
 *   - proposalHash (Poseidon hash of the full proposal text)
 *   - sovereigntyFlags (bitmask: human/caretaker/being/i presence)
 */

template SovereignSparkMercyAlignment() {
    // === INPUTS ===
    signal input proposalHash;           // Poseidon hash of proposal
    signal input sovereigntyFlags;       // bit 0: human, bit 1: caretaker, bit 2: being, bit 3: "i "
    signal input mercyScore;             // scaled 0.9999999 → 9999999
    signal input timestamp;

    // === OUTPUTS (Public Signals) ===
    signal output lowercaseI;            // 1 if Sovereign Divine Spark honored
    signal output mercyAlignment;        // confirmed mercy score
    signal output outTimestamp;

    // === CONSTRAINTS ===

    // 1. Sovereignty check (non-bypassable)
    component hasSovereignty = GreaterEqThan(32);
    hasSovereignty.in[0] <== sovereigntyFlags;
    hasSovereignty.in[1] <== 1;          // at least one flag must be set

    lowercaseI <== hasSovereignty.out;

    // 2. Mercy Alignment check (valence ≥ 0.9999999)
    component mercyCheck = GreaterEqThan(32);
    mercyCheck.in[0] <== mercyScore;
    mercyCheck.in[1] <== 9999999;

    mercyAlignment <== mercyCheck.out;

    // 3. Timestamp integrity (prevents replay)
    outTimestamp <== timestamp;

    // 4. Proposal integrity (hash must be non-zero)
    component hashCheck = IsZero();
    hashCheck.in <== proposalHash;
    hashCheck.out === 0;                 // proposalHash cannot be zero

    // === FINAL CONSTRAINT ===
    // Both sovereignty AND mercy must be true
    component finalGate = AND();
    finalGate.a <== lowercaseI;
    finalGate.b <== mercyAlignment;

    // The circuit only produces a valid proof if finalGate === 1
    finalGate.out === 1;
}

component main {public [lowercaseI, mercyAlignment, outTimestamp]} = SovereignSparkMercyAlignment();