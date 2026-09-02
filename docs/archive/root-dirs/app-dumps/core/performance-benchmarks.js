// performance-benchmarks.js
// Sovereign Performance Benchmark Suite v2026
// Measures end-to-end execution times across all sovereign layers

export class PerformanceBenchmarks {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.runs = 20; // enough for statistical stability
  }

  async runAllBenchmarks() {
    console.log("%c🧪 Running Ra-Thor Performance Benchmarks (20 runs)", "color:#00ff9d; font-size:18px");
    const results = {};

    // Helper: time a function with multiple runs
    const measure = async (name, fn) => {
      const times = [];
      for (let i = 0; i < this.runs; i++) {
        const start = performance.now();
        await fn();
        times.push(performance.now() - start);
      }
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const p95 = times.sort((a,b)=>a-b)[Math.floor(times.length * 0.95)];
      results[name] = { avg: avg.toFixed(2), p95: p95.toFixed(2), unit: "ms" };
      console.log(`  ${name}: avg ${avg.toFixed(2)} ms, p95 ${p95.toFixed(2)} ms`);
    };

    await measure("Full End-to-End Process", async () => {
      await this.orchestrator.process({ rawInput: "benchmark_test", truthFactor: 0.98 });
    });

    await measure("Rust WASM TOLC Proofs", async () => {
      await this.orchestrator.initWasm();
      verify_tolc_convergence(JSON.stringify({ ci: 892 }));
    });

    await measure("Mercy Gates v2 Filtering", async () => {
      this.orchestrator.core.gates.passesAll16Filters({ rawInput: "test" });
    });

    await measure("Lumenas Entropy Corrections", async () => {
      this.orchestrator.core.lumenas.applyHigherOrderEntropyCorrections(892);
    });

    await measure("Ma’at Balance Scoring", async () => {
      this.orchestrator.core.mercyMath.calculateMaAtBalance({ rawInput: "test" });
    });

    await measure("Nilpotent Suppression", async () => {
      this.orchestrator.core.nilpotent.verifySuppression({ rawInput: "test" });
    });

    await measure("Nth-Degree Acceleration", async () => {
      this.orchestrator.core.nthDegree.coforgeInOnePass(892, 892);
    });

    await measure("RBE Simulation Cycle", async () => {
      await this.orchestrator.rbe.simulateCycle({ rawInput: "rbe_test" });
    });

    await measure("WebLLM Mercy Response (fallback)", async () => {
      await this.orchestrator.webllm.generateMercyResponse("benchmark");
    });

    return results;
  }

  async runAndLog() {
    const benchmarks = await this.runAllBenchmarks();
    return benchmarks;
  }
}
