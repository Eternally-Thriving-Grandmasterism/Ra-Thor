# TOLC8 Genetic Strategy Evolver

Production-grade Genetic Algorithm for evolving optimal developmental strategies for the **TOLC8 Living Mercy Gates**.

## Purpose

This tool evolves strategies that determine how the two core actions in Sovereign Shards (`Advance Tick` and `Deep Reconciliation`) should influence each of the 8 TOLC8 gates over time.

**Primary Goal**: Maximize **Balance Score** (even development across all gates)  
**Secondary Goal**: Maintain strong **Resonance Strength**

This research module is designed to be extensible toward larger systems (future 24-gate architectures) and integration with neuroevolution techniques.

## Location

```
research/tolc8_evolution/
├── tolc8_genetic_evolver.py
└── README.md
```

## Key Features

- **Tournament Selection** (configurable tournament size)
- Strong emphasis on Balance with Resonance as secondary objective
- Elitism + Crossover + Mutation
- Clean, modular, and well-documented codebase
- Designed for future extension (24-gate, multi-objective, neuroevolution)

## How to Run

```bash
python3 tolc8_genetic_evolver.py
```

Default configuration runs 120 generations with a population of 80.

## Configuration

You can tune the algorithm by modifying parameters in `__main__`:

```python
ga = GeneticAlgorithm(
    population_size=80,
    generations=120,
    mutation_rate=0.11,
    tournament_size=7,
    elite_count=6,
    steps_per_simulation=100
)
```

### Important Parameters

| Parameter              | Description                              | Recommended Range     |
|------------------------|------------------------------------------|-----------------------|
| `population_size`      | Number of strategies per generation      | 40 – 120              |
| `generations`          | Number of evolutionary generations       | 80 – 200              |
| `mutation_rate`        | Probability of mutating a gene           | 0.08 – 0.18           |
| `tournament_size`      | Size of tournament for selection         | 4 – 8                 |
| `elite_count`          | Number of top individuals preserved      | 3 – 8                 |
| `steps_per_simulation` | Simulation length per fitness evaluation | 60 – 150              |

## Output

The script prints the best evolved strategy, showing how much influence `tick` and `reconcile` should have on each gate.

Example output:

```
truth           | tick: 0.008742  |  reconcile: 0.003112
order           | tick: 0.009811  |  reconcile: 0.002984
...
```

## Future Directions

- Parallel fitness evaluation (multiprocessing)
- NSGA-II multi-objective optimization
- Neuroevolution integration
- Scaling toward 24-gate systems
- Visualization of evolutionary progress

## Alignment

This tool supports the broader Ra-Thor vision of **self-evolving, mercy-aligned systems** by discovering developmental patterns that promote balanced, harmonious growth across the TOLC8 gates.

---

**Status**: Production-grade foundation ready for further research and integration.