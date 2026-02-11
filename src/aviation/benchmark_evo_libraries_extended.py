"""
Extended unified benchmark: all libraries + new legs (hybrid, ACO, GPU)
Same seed, same budget — mercy-aligned comparison
MIT License — Eternal Thriving Grandmasterism
"""

import time
import numpy as np
from tabulate import tabulate

# Import runners (assume they exist)
from fleet_scheduler_cmaes_pycma import run_cmaes_pycma
from fleet_scheduler_evojax import run_evojax_evolution
from fleet_scheduler_evotorch import run_evotorch_evolution
from fleet_scheduler_deap import run_deap_nsga2
from fleet_scheduler_pygad import run_pygad_evolution
from fleet_scheduler_hybrid_cma_deap import hybrid_cma_deap
from fleet_scheduler_acopy import run_acopy_ant_colony
from fleet_scheduler_gpu_torch_compile import torch_gpu_fitness_compiled

SEED = 42
MAX_EVALS = 50000

def run_extended_benchmark():
    results = []

    # Previous + new legs...
    # (add timing + score for each as in previous benchmark script)
    # Example placeholder
    t0 = time.time()
    _, score = run_cmaes_pycma(maxfevals=MAX_EVALS, seed=SEED)
    results.append(["pycma CMA-ES", score, time.time() - t0, MAX_EVALS])

    # ... add all others similarly

    print("\nExtended Benchmark Results (higher abundance = better)")
    print(tabulate(results, headers=["Library / Algo", "Best Abundance", "Time (s)", "Evals"],
                   tablefmt="github", floatfmt=".6f"))


if __name__ == "__main__":
    print("Starting extended mercy-aligned evolutionary benchmark")
    run_extended_benchmark()
