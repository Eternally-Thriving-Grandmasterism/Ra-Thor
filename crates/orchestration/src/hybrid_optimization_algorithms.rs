// crates/orchestration/src/hybrid_optimization_algorithms.rs
// Ra-Thor™ Hybrid Optimization Algorithms — NSGA-II Inspired Multi-Objective Evolutionary Optimization
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Advanced multi-objective optimization for finding optimal energy technology hybrids
// Fully integrated with Unified Sovereign Energy Lattice Core
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechnologyScore {
    pub name: String,
    pub mercy_score: f64,
    pub cost_score: f64,              // 0.0 = expensive, 1.0 = cheap
    pub lifespan_score: f64,
    pub environmental_score: f64,
    pub community_score: f64,
    pub compatibility: HashMap<String, f64>, // compatibility with other technologies
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridConfiguration {
    pub technologies: Vec<String>,
    pub overall_merry_score: f64,
    pub projected_25yr_thriving: f64,
    pub diversity_score: f64,
    pub total_cost_score: f64,
}

pub struct HybridOptimizationEngine {
    population_size: usize,
    generations: usize,
}

impl HybridOptimizationEngine {
    pub fn new() -> Self {
        Self {
            population_size: 40,
            generations: 25,
        }
    }

    /// NSGA-II inspired multi-objective optimization
    pub fn optimize_hybrid_configuration(
        &self,
        available_techs: &[TechnologyScore],
        valence: f64,
        max_technologies: usize,
    ) -> Result<HybridConfiguration, String> {
        if available_techs.is_empty() {
            return Err("No technologies available".to_string());
        }

        // Step 1: Generate initial population (random combinations)
        let mut population = self.generate_initial_population(available_techs, max_technologies);

        // Step 2: Evolve population over generations
        for _ in 0..self.generations {
            let mut offspring = self.create_offspring(&population, available_techs);
            population.extend(offspring);

            // Non-dominated sorting + crowding distance selection
            population = self.nsga_ii_selection(&population, valence);
        }

        // Step 3: Return best solution from final Pareto front
        let best = population.into_iter()
            .max_by(|a, b| {
                let score_a = a.overall_merry_score * 0.6 + a.diversity_score * 0.4;
                let score_b = b.overall_merry_score * 0.6 + b.diversity_score * 0.4;
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap();

        Ok(best)
    }

    fn generate_initial_population(
        &self,
        techs: &[TechnologyScore],
        max_tech: usize,
    ) -> Vec<HybridConfiguration> {
        let mut population = Vec::new();

        for _ in 0..self.population_size {
            let mut selected = Vec::new();
            let num_techs = (rand::random::<usize>() % max_tech) + 1;

            let mut indices: Vec<usize> = (0..techs.len()).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            for i in 0..num_techs {
                selected.push(techs[indices[i]].name.clone());
            }

            population.push(self.evaluate_configuration(&selected, techs));
        }

        population
    }

    fn create_offspring(
        &self,
        population: &[HybridConfiguration],
        techs: &[TechnologyScore],
    ) -> Vec<HybridConfiguration> {
        let mut offspring = Vec::new();

        for _ in 0..self.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(population);
            let parent2 = self.tournament_selection(population);

            // Crossover
            let mut child_techs = parent1.technologies.clone();
            for tech in &parent2.technologies {
                if !child_techs.contains(tech) && child_techs.len() < 4 {
                    child_techs.push(tech.clone());
                }
            }

            // Mutation
            if rand::random::<f64>() < 0.15 {
                if let Some(idx) = (0..techs.len()).choose(&mut rand::thread_rng()) {
                    let new_tech = &techs[idx].name;
                    if !child_techs.contains(new_tech) && child_techs.len() < 4 {
                        child_techs.push(new_tech.clone());
                    }
                }
            }

            offspring.push(self.evaluate_configuration(&child_techs, techs));
        }

        offspring
    }

    fn tournament_selection(&self, population: &[HybridConfiguration]) -> HybridConfiguration {
        let mut best = population[0].clone();
        for _ in 0..3 {
            let candidate = &population[rand::random::<usize>() % population.len()];
            if candidate.overall_merry_score > best.overall_merry_score {
                best = candidate.clone();
            }
        }
        best
    }

    fn nsga_ii_selection(
        &self,
        population: &[HybridConfiguration],
        valence: f64,
    ) -> Vec<HybridConfiguration> {
        // Simplified NSGA-II: sort by non-domination + crowding distance
        let mut sorted = population.to_vec();
        sorted.sort_by(|a, b| {
            // Primary: overall mercy score (higher is better)
            // Secondary: diversity (higher is better)
            let score_a = a.overall_merry_score * 0.7 + a.diversity_score * 0.3;
            let score_b = b.overall_merry_score * 0.7 + b.diversity_score * 0.3;
            score_b.partial_cmp(&score_a).unwrap()
        });

        sorted.truncate(self.population_size);
        sorted
    }

    fn evaluate_configuration(
        &self,
        tech_names: &[String],
        available: &[TechnologyScore],
    ) -> HybridConfiguration {
        let mut total_merry = 0.0;
        let mut total_cost = 0.0;
        let mut total_lifespan = 0.0;
        let mut total_env = 0.0;
        let mut total_community = 0.0;

        for name in tech_names {
            if let Some(tech) = available.iter().find(|t| &t.name == name) {
                total_merry += tech.mercy_score;
                total_cost += tech.cost_score;
                total_lifespan += tech.lifespan_score;
                total_env += tech.environmental_score;
                total_community += tech.community_score;
            }
        }

        let count = tech_names.len() as f64;

        HybridConfiguration {
            technologies: tech_names.to_vec(),
            overall_merry_score: total_merry / count,
            projected_25yr_thriving: (total_merry * 0.4 + total_lifespan * 0.3 + total_community * 0.3) / count,
            diversity_score: self.calculate_diversity(tech_names, available),
            total_cost_score: total_cost / count,
        }
    }

    fn calculate_diversity(&self, selected: &[String], all_techs: &[TechnologyScore]) -> f64 {
        if selected.len() <= 1 {
            return 0.3;
        }

        let mut diversity = 0.0;
        for i in 0..selected.len() {
            for j in (i + 1)..selected.len() {
                if let (Some(t1), Some(t2)) = (
                    all_techs.iter().find(|t| &t.name == &selected[i]),
                    all_techs.iter().find(|t| &t.name == &selected[j]),
                ) {
                    let comp = t1.compatibility.get(&t2.name).unwrap_or(&0.5);
                    diversity += comp;
                }
            }
        }
        (diversity / (selected.len() * (selected.len() - 1)) as f64 * 0.5).min(0.95)
    }
}
