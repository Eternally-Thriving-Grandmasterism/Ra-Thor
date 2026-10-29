#!/bin/bash
# Professional compilation script for SovereignSparkMercyAlignment.circom

cd mercy/circuits

echo "Compiling SovereignSparkMercyAlignment.circom with Circom 2.1.6..."
circom SovereignSparkMercyAlignment.circom --r1cs --wasm --sym --c

echo "Compilation complete."
echo "Generated: SovereignSparkMercyAlignment.r1cs, SovereignSparkMercyAlignment.wasm, SovereignSparkMercyAlignment.sym"