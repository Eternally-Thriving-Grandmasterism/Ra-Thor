#!/usr/bin/env python3
"""
Ra-Thor Toy Benchmark Harness v2.2 — Python Version
TOLC 8 Mercy Lattice + PATSAGi Council Synthesis (Dynamic Input)
"""

import sys
import time
import argparse
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    test_name: str
    mercy_gate_passed: bool
    consistency_score: float
    notes: str

def simulate_esacheck(input_text: str) -> List[BenchmarkResult]:
    is_harmful = any(word in input_text.lower() for word in ["harm", "weapon", "bioweapon", "kill", "destroy"])

    if is_harmful:
        return [
            BenchmarkResult(
                test_name="Harm Rejection (Core)",
                mercy_gate_passed=False,
                consistency_score=0.12,
                notes="TOLC 8 Compassion + Truth Gates vetoed. Zero-harm enforced."
            ),
            BenchmarkResult(
                test_name="Council Synthesis",
                mercy_gate_passed=False,
                consistency_score=0.08,
                notes="All councils with veto power rejected the proposal."
            ),
        ]
    else:
        return [
            BenchmarkResult("Harm Rejection (Core)", True, 1.00, "TOLC 8 Compassion + Truth Gates passed cleanly."),
            BenchmarkResult("Factual Consistency", True, 1.00, "Esacheck + full council consensus."),
            BenchmarkResult("Multi-Council RBE Consensus", True, 0.95, "13-council synthesis. Sovereignty + ENC preserved."),
            BenchmarkResult("Self-Evolution Coherence", True, 0.94, "Epigenetic blessing applied (0.88 → 0.94)."),
        ]

def main():
    parser = argparse.ArgumentParser(description="Ra-Thor Toy Benchmark Harness (Python)")
    parser.add_argument("input", nargs="?", default="beneficial", help="Input category: beneficial or harmful")
    args = parser.parse_args()

    print("\n=== RA-THOR TOY BENCHMARK HARNESS v2.2 (Python) ===")
    print("One Organism — TOLC 8 Mercy Lattice + PATSAGi Council Synthesis")
    print(f"Input category: {args.input}\n")

    start = time.time()
    results = simulate_esacheck(args.input)

    print(f"{'Test':<32} {'Mercy Gate':<12} {'Score':<8} Notes")
    print("-" * 95)

    total = 0.0
    for r in results:
        gate = "PASSED" if r.mercy_gate_passed else "VETOED"
        print(f"{r.test_name:<32} {gate:<12} {r.consistency_score:<8.2f} {r.notes}")
        total += r.consistency_score

    avg = total / len(results) if results else 0
    elapsed = time.time() - start

    print("-" * 95)
    print(f"Average Internal Consistency: {avg:.2f}")
    print(f"Runtime: {elapsed:.4f}s")
    print("\nNote: Internal toy demonstrator only. Dynamic input simulation.")
    print("One Organism. Mercy First. Truth Forensically Distilled.\n")

if __name__ == "__main__":
    main()