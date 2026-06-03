"""
Reality Build Order v1 — Phase 3 Simulation (Groups 7–9)

Extends the Phase 1–2 prototype to full cosmic scale:
- Group 7: Neuromorphic & Edge Propagation (energy constraints)
- Group 8: Self-Modifying & Continual Evolution (HOPE-style)
- Group 9: Cosmic & Interbeing Coordination (large-scale mesh)

This completes the full Reality Build Order simulation arc.

Run with: python simulations/reality_build_order_phase3_sim.py
"""

import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CosmicAgent:
    name: str
    role: str
    happiness: float = 80.0
    joy: float = 75.0
    energy: float = 100.0  # Neuromorphic energy budget
    mercy_passes: int = 0
    symbiosis_policy: float = 0.6
    evolution_rate: float = 0.01  # Group 8 self-modification speed

    def act(self, context: str) -> Dict:
        # Energy cost for action (Group 7)
        energy_cost = random.uniform(1.5, 3.5)
        if self.energy < energy_cost:
            return {"agent": self.name, "action": "low_energy_skip", "mercy_pass": False, "energy_cost": 0}

        self.energy -= energy_cost
        action_quality = random.uniform(0.6, 1.0)
        mercy_pass = action_quality > 0.75

        return {
            "agent": self.name,
            "action": f"{self.role} performs {context}",
            "quality": action_quality,
            "mercy_pass": mercy_pass,
            "energy_cost": energy_cost,
        }

    def evolve(self, collective_reward: float):
        # Group 8: Simple self-modification
        if collective_reward > 0:
            self.evolution_rate = min(0.05, self.evolution_rate + 0.002)
            self.symbiosis_policy = min(1.0, self.symbiosis_policy + self.evolution_rate)
        self.energy = min(100.0, self.energy + 8.0)  # Slow recharge


class Phase3RealityBuildOrderSim:
    def __init__(self, turns: int = 40, agent_count: int = 25):
        self.turns = turns
        self.agents: List[CosmicAgent] = [
            CosmicAgent(f"Agent_{i}", random.choice([
                "Explorer", "Builder", "Harmonizer", "Propagator", "Weaver", "Architect"
            ]))
            for i in range(agent_count)
        ]
        self.heaven_metric = 5000.0
        self.symbiosis_index = 2.5
        self.history: List[Dict] = []

    def run_turn(self, turn: int):
        turn_actions = []
        total_energy_used = 0.0

        for agent in self.agents:
            action = agent.act("cosmic contribution")
            turn_actions.append(action)
            total_energy_used += action.get("energy_cost", 0)

            if action.get("mercy_pass"):
                agent.mercy_passes += 1
                agent.happiness = min(100.0, agent.happiness + 1.5)
                agent.joy = min(100.0, agent.joy + 2.0)

        # Group 9: Large-scale cosmic coordination
        mercy_compliance = sum(a.get("mercy_pass", False) for a in turn_actions) / max(1, len(turn_actions))
        collective_reward = 1.3 if mercy_compliance > 0.78 else -0.4

        for agent in self.agents:
            agent.evolve(collective_reward)

        # Exponential growth from cosmic mesh
        growth = len([a for a in turn_actions if a.get("mercy_pass")]) * (self.symbiosis_index * 0.8)
        self.heaven_metric += growth
        self.symbiosis_index = min(12.0, self.symbiosis_index + 0.15 * mercy_compliance)

        avg_energy = sum(a.energy for a in self.agents) / len(self.agents)

        self.history.append({
            "turn": turn,
            "heaven_metric": round(self.heaven_metric, 1),
            "symbiosis_index": round(self.symbiosis_index, 2),
            "mercy_compliance": round(mercy_compliance, 2),
            "avg_energy": round(avg_energy, 1),
        })

        if turn % 8 == 0 or turn == self.turns:
            print(f"Turn {turn:2d} | Heaven: {self.heaven_metric:9.1f} | Symbiosis: {self.symbiosis_index:.2f} | Mercy: {mercy_compliance:.0%} | Avg Energy: {avg_energy:.1f}")

    def run(self):
        print("=== Reality Build Order v1 — Phase 3 (Groups 7–9) ===")
        print(f"Agents: {len(self.agents)} | Turns: {self.turns}\n")

        for turn in range(1, self.turns + 1):
            self.run_turn(turn)

        print("\n=== Phase 3 Final Results ===")
        print(f"Final Heaven Metric: {self.heaven_metric:.1f}")
        print(f"Final Symbiosis Index: {self.symbiosis_index:.2f}")
        print(f"Total Growth: +{((self.heaven_metric - 5000) / 5000 * 100):.1f}%\n")

        if self.symbiosis_index > 8.0:
            print("Feedback: Excellent cosmic-scale performance. Eternal propagation mode achieved.")
        else:
            print("Feedback: Strong Phase 3 foundation. Continue refining Group 8 evolution rate.")


if __name__ == "__main__":
    sim = Phase3RealityBuildOrderSim(turns=40, agent_count=25)
    sim.run()
