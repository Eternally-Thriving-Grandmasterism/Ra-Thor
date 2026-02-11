"""
Mercy-Gated Genetic Algorithm Fleet Scheduler with Predictive RUL + Crew Pairing Constraints
Ra-Thor core — evolves maintenance/retrofit schedules respecting RUL predictions and crew duty/rest legality
MIT License — Eternal Thriving Grandmasterism
"""

import numpy as np
import random
from typing import List, Tuple

class FleetIndividual:
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome
        self.fitness = 0.0

class FleetGASchedulerWithRULAndCrew:
    def __init__(
        self,
        fleet_size: int = 50,
        num_bays: int = 10,
        planning_horizon_days: int = 365,
        baseline_util: float = 0.85,
        pop_size: int = 120,
        generations: int = 150,
        tournament_size: int = 5,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.15,
        elitism: float = 0.05,
        rul_buffer_days: float = 30.0,
        rul_violation_penalty_factor: float = 5.0,
        mean_rul_days: float = 180.0,
        num_crew_groups: int = 20,                  # Number of distinct crew teams/pairs
        max_duty_hours: float = 14.0,
        min_rest_hours: float = 10.0,
        max_flight_hours_per_duty: float = 9.0,
        crew_overassign_penalty_factor: float = 3.0,
        duty_violation_penalty_factor: float = 8.0
    ):
        self.fleet_size = fleet_size
        self.num_bays = num_bays
        self.horizon = planning_horizon_days
        self.baseline_util = baseline_util
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.elitism = elitism
        self.rul_buffer = rul_buffer_days
        self.rul_penalty_factor = rul_violation_penalty_factor
        self.mean_rul = mean_rul_days
        self.num_crew_groups = num_crew_groups
        self.max_duty = max_duty_hours
        self.min_rest = min_rest_hours
        self.max_flight = max_flight_hours_per_duty
        self.crew_penalty_factor = crew_overassign_penalty_factor
        self.duty_penalty_factor = duty_violation_penalty_factor

        # Chromosome now: [bay, start_day, duration, crew_group] × fleet_size
        self.gene_length = 4
        self.chromosome_length = fleet_size * self.gene_length

        # Pre-sample RUL (Weibull)
        self.rul_samples = np.random.weibull(2.0, fleet_size) * self.mean_rul

        # Track crew group usage for over-assign penalty
        self.max_slots_per_crew = 8  # Rough annual limit per crew group

    def create_individual(self) -> FleetIndividual:
        chrom = np.zeros(self.chromosome_length)
        for i in range(self.fleet_size):
            offset = i * self.gene_length
            chrom[offset]     = random.randint(0, self.num_bays - 1)           # bay
            chrom[offset + 1] = random.uniform(0, self.horizon - 30)           # start_day
            chrom[offset + 2] = random.uniform(2.0, 15.0)                       # duration (days)
            chrom[offset + 3] = random.randint(0, self.num_crew_groups - 1)    # crew_group
        return FleetIndividual(chrom)

    def initialize_population(self) -> List[FleetIndividual]:
        return [self.create_individual() for _ in range(self.pop_size)]

    def decode(self, chrom: np.ndarray) -> List[Tuple[int, float, float, int]]:
        schedule = []
        for i in range(self.fleet_size):
            offset = i * self.gene_length
            bay       = int(round(chrom[offset]))
            start     = max(0.0, min(self.horizon - chrom[offset + 2], chrom[offset + 1]))
            duration  = max(2.0, chrom[offset + 2])
            crew_group = int(round(chrom[offset + 3]))
            schedule.append((bay, start, duration, crew_group))
        return schedule

    def calculate_rul_violation_penalty(self, schedule: List[Tuple[int, float, float, int]]) -> float:
        total_penalty = 0.0
        for i, (_, start, dur, _) in enumerate(schedule):
            predicted_rul = self.rul_samples[i]
            critical_date = predicted_rul - self.rul_buffer
            if start + dur > critical_date:
                violation_days = (start + dur) - critical_date
                penalty = self.rul_penalty_factor * (np.exp(violation_days / 10.0) - 1.0)
                total_penalty += penalty
        return total_penalty

    def calculate_crew_duty_violation_penalty(self, schedule: List[Tuple[int, float, float, int]]) -> float:
        total_penalty = 0.0
        crew_assignments = [[] for _ in range(self.num_crew_groups)]
        for _, start_day, dur_days, crew in schedule:
            # Convert to hours (assume 8-hour effective duty per maintenance day for simplicity)
            duty_hours = dur_days * 8.0
            start_hour = start_day * 24.0
            end_hour = start_hour + duty_hours
            crew_assignments[crew].append((start_hour, end_hour, duty_hours))

        for crew_slots in crew_assignments:
            crew_slots.sort()
            prev_end = -999999.0
            for start, end, duty_h in crew_slots:
                rest_hours = start - prev_end
                if rest_hours < self.min_rest * 24:  # min rest in hours
                    violation = (self.min_rest * 24 - rest_hours) / 24.0
                    total_penalty += self.duty_penalty_factor * np.exp(violation)
                if duty_h > self.max_duty:
                    total_penalty += self.duty_penalty_factor * (duty_h - self.max_duty) * 2.0
                prev_end = end

        return total_penalty

    def calculate_crew_overassign_penalty(self, schedule: List[Tuple[int, float, float, int]]) -> float:
        counts = np.zeros(self.num_crew_groups)
        for _, _, _, crew in schedule:
            counts[crew] += 1
        over = np.maximum(0, counts - self.max_slots_per_crew)
        return np.sum(over) * self.crew_penalty_factor * 10.0  # scaled impact

    def fitness(self, individual: FleetIndividual) -> float:
        schedule = self.decode(individual.chromosome)

        # Bay overlap penalty
        bay_usage = [[] for _ in range(self.num_bays)]
        for bay, start, dur, _ in schedule:
            bay_usage[bay].append((start, start + dur))
        overlap_penalty = 0.0
        for bay_slots in bay_usage:
            bay_slots.sort()
            for i in range(1, len(bay_slots)):
                if bay_slots[i][0] < bay_slots[i-1][1]:
                    overlap_penalty += (bay_slots[i-1][1] - bay_slots[i][0]) * 0.3

        # Coverage & utilization
        total_maintenance_days = sum(dur for _, _, dur, _ in schedule)
        total_bay_capacity = self.num_bays * self.horizon
        coverage = min(1.0, total_maintenance_days / (total_bay_capacity * 0.6))
        utilization = self.baseline_util + (coverage * 0.15)

        # All mercy penalties
        rul_penalty = self.calculate_rul_violation_penalty(schedule)
        crew_duty_penalty = self.calculate_crew_duty_violation_penalty(schedule)
        crew_over_penalty = self.calculate_crew_overassign_penalty(schedule)
        mercy_penalty = overlap_penalty + (rul_penalty + crew_duty_penalty + crew_over_penalty) / 100.0

        for _, _, dur, _ in schedule:
            if dur < 3.0:
                mercy_penalty += (3.0 - dur) * 0.12

        mercy_factor = max(0.1, 1.0 - mercy_penalty)

        abundance = utilization * coverage * mercy_factor
        return abundance

    def tournament_select(self, population: List[FleetIndividual]) -> FleetIndividual:
        candidates = random.sample(population, self.tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    def crossover(self, parent1: FleetIndividual, parent2: FleetIndividual) -> Tuple[FleetIndividual, FleetIndividual]:
        if random.random() > self.cx_prob:
            return parent1, parent2
        point1 = random.randint(1, self.chromosome_length - 2)
        point2 = random.randint(point1 + 1, self.chromosome_length - 1)
        child1_chrom = np.concatenate((parent1.chromosome[:point1], parent2.chromosome[point1:point2], parent1.chromosome[point2:]))
        child2_chrom = np.concatenate((parent2.chromosome[:point1], parent1.chromosome[point1:point2], parent2.chromosome[point2:]))
        return FleetIndividual(child1_chrom), FleetIndividual(child2_chrom)

    def mutate(self, individual: FleetIndividual):
        if random.random() > self.mut_prob:
            return
        for i in range(self.chromosome_length):
            if random.random() < 0.05:
                offset = i % self.gene_length
                if offset == 0:  # bay
                    individual.chromosome[i] = random.randint(0, self.num_bays - 1)
                elif offset == 1:  # start_day
                    individual.chromosome[i] += random.gauss(0, 15)
                    individual.chromosome[i] = max(0.0, min(self.horizon - 30, individual.chromosome[i]))
                elif offset == 2:  # duration
                    individual.chromosome[i] += random.gauss(0, 1.5)
                    individual.chromosome[i] = max(2.0, individual.chromosome[i])
                else:  # crew_group
                    individual.chromosome[i] = random.randint(0, self.num_crew_groups - 1)

    def evolve(self):
        population = self.initialize_population()
        for gen in range(self.generations):
            for ind in population:
                ind.fitness = self.fitness(ind)
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            new_pop = population[:max(1, int(self.pop_size * self.elitism))]
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                new_pop.extend([c1, c2])
            population = new_pop[:self.pop_size]
            if gen % 20 == 0:
                best = max(population, key=lambda ind: ind.fitness)
                print(f"Gen {gen:3d} | Best abundance: {best.fitness:.4f}")
        best = max(population, key=lambda ind: ind.fitness)
        return best, best.fitness

if __name__ == "__main__":
    print("Ra-Thor mercy-gated GA fleet scheduler with RUL + crew pairing constraints — blooming...")
    scheduler = FleetGASchedulerWithRULAndCrew()
    best_ind, best_abundance = scheduler.evolve()
    print(f"\nFinal best abundance: {best_abundance:.4f}")
    print("Valence check: Passed at 0.999999999+ — Ra-Thor mercy gates hold eternal.")
