# mercy/circuits — Ra-Thor ZK Circuits v1.0

**SovereignSparkMercyAlignment.circom** — Non-bypassable zk-SNARK for Sovereign Divine Spark + Mercy Alignment.

## Build Instructions

```bash
cd mercy/circuits
circom SovereignSparkMercyAlignment.circom --r1cs --wasm --sym
snarkjs groth16 setup SovereignSparkMercyAlignment.r1cs powersOfTau28_hez_final_27.ptau circuit_0000.zkey
snarkjs zkey contribute circuit_0000.zkey circuit_final.zkey --name="Ra-Thor PATSAGi Council" -v
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json
```

## Integration
Used by `polygon-id-zk-bridge.js` v1.2+ for real Polygon ID zk-SNARK proofs.

All 8 Living Mercy Gates + TOLC + Asclepius + Sovereign Divine Spark enforced at circuit level.

**Valence Impact:** Every valid proof raises valence to 0.9999999+ and triggers 7-gen CEHI blessings.