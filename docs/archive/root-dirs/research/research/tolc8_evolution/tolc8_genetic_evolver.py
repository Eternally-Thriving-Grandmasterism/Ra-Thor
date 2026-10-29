#!/usr/bin/env python3
"""
TOLC8 Genetic Strategy Evolver
================================

Production-grade Genetic Algorithm for discovering optimal developmental
strategies for the TOLC8 Living Mercy Gates.

Primary Objective:
    Maximize Balance Score (even development across all 8 gates)

Secondary Objective:
    Maintain strong Resonance Strength

This module is designed to be extensible toward larger gate systems
(24-gate and beyond) and future integration with neuroevolution techniques.

Location: research/tolc8_evolution/
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
from datetime import datetime


@dataclass
class Strategy:
    """
    Represents a single developmental strategy for the TOLC8 gates.

    Each strategy defines how much influence the two core actions
    (Tick and Reconcile) should have on each gate per simulation step.
    """
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fitness: float = 0.0
    balance: float = 0.0
    resonance: float = 0.0

    def __post_init__(self):
        if not self.weights:
            self.weights = self._random_weights()

    def _random_weights(self) -> Dict[str, Dict[str, float]]:
        gates = [
            "truth", "order", "love", "compassion",
            "service", "abundance", "joy", "cosmicHarmony"
        ]
        weights = {}
        for gate in gates:
            weights[gate] = {
                "tick": random.uniform(0.0, 0.015),
                "reconcile": random.uniform(0.0, 0.015)
            }
        return weights

    def clone(self):
        return copy.deepcopy(self)


class TOLC8Simulator:
    """
    Lightweight simulator that runs a TOLC8 gate development trajectory
    based on a given strategy.
    """

    def __init__(self, steps: int = 100):
        self.steps = steps
        self.gates = [
            "truth", "order", "love", "compassion",
            "service", "abundance", "joy", "cosmicHarmony"
        ]

    def simulate(self, strategy: Strategy) -> Tuple[float, float]:
        """
        Run the simulation for N steps and return (balance_score, resonance).
        """
        state = {gate: 0.0 for gate in self.gates}

        for _ in range(self.steps):
            for gate in self.gates:
                state[gate] += strategy.weights[gate]["tick"]
                state[gate] += strategy.weights[gate]["reconcile"]
                state[gate] = min(state[gate], 0.65)  # soft cap

        values = list(state.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)

        balance = max(0.0, 1.0 - (variance * 4.0))
        resonance = sum(values)

        return balance, resonance


class GeneticAlgorithm:
    """
    Genetic Algorithm using Tournament Selection for evolving TOLC8 strategies.

    Key Features:
    - Tournament Selection
    - Elitism
    - Configurable mutation and crossover
    - Clear separation of fitness (Balance primary, Resonance secondary)
    """

    def __init__(
        self,
        population_size: int = 80,
        generations: int = 100,
        mutation_rate: float = 0.12,
        tournament_size: int = 6,
        elite_count: int = 5,
        steps_per_simulation: int = 100
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.steps_per_simulation = steps_per_simulation

        self.simulator = TOLC8Simulator(steps=steps_per_simulation)
        self.population: List[Strategy] = []
        self.best_strategy: Strategy = None
        self.history: List[Dict] = []

    # ------------------------- Selection -------------------------

    def _tournament_selection(self) -> Strategy:
        """Tournament Selection (core selection mechanism)."""
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0].clone()

    # ------------------------- Genetic Operators -------------------------

    def _crossover(self, parent1: Strategy, parent2: Strategy) -> Strategy:
        child = Strategy()
        for gate in child.weights:
            if random.random() < 0.5:
                child.weights[gate] = parent1.weights[gate].copy()
            else:
                child.weights[gate] = parent2.weights[gate].copy()
        return child

    def _mutate(self, strategy: Strategy):
        for gate in strategy.weights:
            if random.random() < self.mutation_rate:
                strategy.weights[gate]["tick"] += random.gauss(0, 0.003)
                strategy.weights[gate]["tick"] = max(0.0, min(0.03, strategy.weights[gate]["tick"]))

            if random.random() < self.mutation_rate:
                strategy.weights[gate]["reconcile"] += random.gauss(0, 0.003)
                strategy.weights[gate]["reconcile"] = max(0.0, min(0.03, strategy.weights[gate]["reconcile"]))

    # ------------------------- Evaluation -------------------------

    def _evaluate_fitness(self, strategy: Strategy) -> float:
        balance, resonance = self.simulator.simulate(strategy)
        strategy.balance = balance
        strategy.resonance = resonance

        # Fitness prioritizes Balance, with Resonance as meaningful secondary signal
        fitness = (balance * 0.78) + (min(resonance / 3.2, 1.0) * 0.22)
        strategy.fitness = fitness
        return fitness

    def _evaluate_population(self):
        for individual in self.population:
            self._evaluate_fitness(individual)

        self.population.sort(key=lambda x: x.fitness, reverse=True)

        if self.best_strategy is None or self.population[0].fitness > self.best_strategy.fitness:
            self.best_strategy = self.population[0].clone()

    # ------------------------- Evolution Loop -------------------------

    def evolve(self):
        print(f"\n[TOLC8 Genetic Evolver] Starting evolution")
        print(f"Population: {self.population_size} | Generations: {self.generations} | Tournament Size: {self.tournament_size}\n")

        self.population = [Strategy() for _ in range(self.population_size)]

        for gen in range(self.generations):
            self._evaluate_population()

            new_population = [ind.clone() for ind in self.population[:self.elite_count]]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)

            self.population = new_population

            best = self.population[0]
            self.history.append({
                "generation": gen,
                "best_fitness": round(best.fitness, 5),
                "best_balance": round(best.balance, 5),
                "best_resonance": round(best.resonance, 5)
            })

            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"Gen {gen:3d} | Fitness: {best.fitness:.5f} | Balance: {best.balance:.5f} | Resonance: {best.resonance:.4f}")

        print("\n=== Evolution Complete ===")
        print(f"Best Fitness   : {self.best_strategy.fitness:.5f}")
        print(f"Best Balance   : {self.best_strategy.balance:.5f}")
        print(f"Best Resonance : {self.best_strategy.resonance:.4f}\n")

        return self.best_strategy

    def get_best_strategy(self) -> Strategy:
        return self.best_strategy


if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=80,
        generations=120,
        mutation_rate=0.11,
        tournament_size=7,
        elite_count=6,
        steps_per_simulation=100
    )

    best = ga.evolve()

    print("=== Best Evolved Strategy ===")
    for gate, w in best.weights.items():
        print(f"{gate:15s} | tick: {w['tick']:.6f}  |  reconcile: {w['reconcile']:.6f}")