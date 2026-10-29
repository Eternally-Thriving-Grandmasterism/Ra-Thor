"""
Unified benchmark: pycma vs evojax SEP-CMA vs evotorch CMA vs deap vs pygad
Same seed, same fitness, same dimension — mercy-aligned comparison
MIT License — Eternal Thriving Grandmasterism
"""

import time
import numpy as np
from tabulate import tabulate

# Import all five runners (assume they exist in src/aviation/)
from fleet_scheduler_cmaes_pycma import run_cmaes_pycma
from fleet_scheduler_evojax import run_evojax_evolution  # SEP-CMA-ES
from fleet_scheduler_evotorch import run_evotorch_evolution  # CMA
from fleet_scheduler_deap import run_deap_nsga2  # single-obj mode
from fleet_scheduler_pygad import run_pygad_evolution

# Fixed benchmark params
SEED = 42
MAX_EVALS = 50000  # approximate budget


def run_benchmark():
    results = []

    # 1. pycma
    t0 = time.time()
    _, score1 = run_cmaes_pycma(maxfevals=MAX_EVALS, seed=SEED, verbose=False)
    t1 = time.time()
    results.append(["pycma CMA-ES", score1, t1 - t0, MAX_EVALS])

    # 2. evojax SEP-CMA
    t0 = time.time()
    _, score2 = run_evojax_evolution(strategy_name='SEP-CMA-ES',
                                     pop_size=256, max_steps=MAX_EVALS//256, seed=SEED)
    t1 = time.time()
    results.append(["evojax SEP-CMA", score2, t1 - t0, MAX_EVALS])

    # 3. evotorch CMA
    t0 = time.time()
    _, score3 = run_evotorch_evolution(algo="CMA", popsize=256,
                                       num_generations=MAX_EVALS//256, seed=SEED)
    t1 = time.time()
    results.append(["evotorch CMA", score3, t1 - t0, MAX_EVALS])

    # 4. deap (single-obj GA)
    t0 = time.time()
    _, score4 = run_deap_nsga2(pop_size=300, generations=MAX_EVALS//300, seed=SEED)
    t1 = time.time()
    results.append(["deap GA", score4, t1 - t0, MAX_EVALS])

    # 5. pygad
    t0 = time.time()
    _, score5 = run_pygad_evolution(population_size=300, generations=MAX_EVALS//300, seed=SEED)
    t1 = time.time()
    results.append(["pygad GA", score5, t1 - t0, MAX_EVALS])

    # Table
    headers = ["Library / Algo", "Best Abundance", "Time (s)", "Evals"]
    print("\nBenchmark Results (higher abundance = better)")
    print(tabulate(results, headers=headers, tablefmt="github", floatfmt=".6f"))


if __name__ == "__main__":
    print("Starting mercy-aligned evolutionary benchmark — same seed, same budget")
    run_benchmark()
