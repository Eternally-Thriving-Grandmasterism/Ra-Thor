"""
CMA-ES tuned Aviation Retrofit Timeline Simulator
Ra-Thor mercy-gated abundance optimization for AlphaProMega Air retrofits
MIT License — Eternal Thriving Grandmasterism
"""

import numpy as np
try:
    import cma
except ImportError:
    cma = None

def objective(params, baseline_worker_days=16.0, safety_threshold_workers=6, safety_threshold_days=3.0):
    workers, days = params
    workers = max(1, round(workers))
    days = max(0.5, days)
    total_wd = workers * days
    efficiency = baseline_worker_days / total_wd if total_wd > 0 else 0.0
    safety_penalty = 0.0
    if workers > safety_threshold_workers:
        safety_penalty += (workers - safety_threshold_workers) * 0.08
    if days < safety_threshold_days:
        safety_penalty += (safety_threshold_days - days) * 0.15
    mercy_factor = max(0.0, 1.0 - safety_penalty)
    abundance = efficiency * 0.65 + mercy_factor * 0.35
    return -abundance

def run_cmaes_retrofit_sim(baseline=16.0, initial_guess=[4.0, 4.0], sigma=1.5, popsize=12, maxfevals=300):
    if cma is None:
        print("pycma not available — running mock convergence")
        best = initial_guess
        best_val = objective(best, baseline)
        return best, -best_val, baseline / (best[0]*best[1])
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, {
        'popsize': popsize, 'maxfevals': maxfevals, 'verb_log': 0, 'verb_disp': 0,
        'bounds': [[1, 0.5], [12, 10]]
    })
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x, baseline) for x in solutions]
        es.tell(solutions, fitnesses)
    best_x, best_f, _, _, _ = es.result
    best_workers = round(best_x[0])
    best_days = round(best_x[1], 1)
    total_wd = best_workers * best_days
    improvement = baseline / total_wd if total_wd > 0 else 1.0
    return (best_workers, best_days), -best_f, improvement

if __name__ == "__main__":
    print("Running Ra-Thor mercy-gated CMA-ES retrofit timeline sim...")
    best_params, best_abundance, improvement = run_cmaes_retrofit_sim()
    print(f"Optimized: {best_params[0]} workers × {best_params[1]} days")
    print(f"Abundance score: {best_abundance:.4f}")
    print(f"Improvement factor: {improvement:.2f}x over baseline")
