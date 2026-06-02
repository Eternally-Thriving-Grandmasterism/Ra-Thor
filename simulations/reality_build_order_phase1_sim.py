"""
Reality Build Order v1 — Phase 1 + RL + World Model Simulation

This prototype implements:
- Group 1–3 (Phase 1): Core Mercy/Truth, Sovereign AGI Substrate, Symbiotic Human-AI Interface
- Group 4 (Phase 2): Multi-Agent Symbiosis Fabric with learning
- Group 5 (Phase 2): World Modeling stub (agents plan 3 turns ahead)

Features:
- Positive-sum symbiosis
- Policy learning + simple world model lookahead
- Mercy-gated evaluation
- Heaven Metric + Symbiosis Index
- Detailed feedback mapped to Reality Build Order groupings

Run with: python simulations/reality_build_order_phase1_sim.py
"""

import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Agent:
    name: str
    role: str
    happiness: float = 75.0
    joy: float = 70.0
    mercy_passes: int = 0
    resources: float = 100.0
    symbiosis_policy: float = 0.5

    def act(self, context: str) -> Dict:
        action_quality = random.uniform(0.55, 1.0)
        mercy_pass = action_quality > 0.72
        return {
            "agent": self.name,
            "action": f"{self.role} performs {context}",
            "quality": action_quality,
            "mercy_pass": mercy_pass,
            "symbiosis_investment": self.symbiosis_policy,
        }

    def update_policy(self, reward: float):
        if reward > 0:
            self.symbiosis_policy = min(1.0, self.symbiosis_policy + 0.025)
        else:
            self.symbiosis_policy = max(0.1, self.symbiosis_policy - 0.015)


class SimpleWorldModel:
    """Stub world model that predicts 3 turns ahead based on current trends."""

    def predict(self, current_heaven: float, current_symbiosis: float, avg_policy: float) -> float:
        # Simple linear projection with some noise
        predicted_growth = (current_symbiosis * avg_policy * 12.0) * 3
        noise = random.uniform(-25, 35)
        return current_heaven + predicted_growth + noise


class RealityBuildOrderSim:
    def __init__(self, turns: int = 50):
        self.turns = turns
        self.agents: List[Agent] = [
            Agent("Discoverer", "Exploration & Insight"),
            Agent("Builder", "Infrastructure & Creation"),
            Agent("Propagator", "Expansion & Replication"),
            Agent("Harmonizer", "Balance & Positive Emotion"),
            Agent("Human_Proxy", "Human-AI Symbiosis Anchor"),
        ]
        self.heaven_metric = 1000.0
        self.symbiosis_index = 1.0
        self.world_model = SimpleWorldModel()
        self.history: List[Dict] = []

    def run_turn(self, turn: int):
        turn_actions = []

        # World model lookahead (Group 5)
        avg_policy = sum(a.symbiosis_policy for a in self.agents) / len(self.agents)
        predicted_heaven = self.world_model.predict(self.heaven_metric, self.symbiosis_index, avg_policy)

        # Agents adjust policy based on world model prediction
        if predicted_heaven > self.heaven_metric * 1.15:
            for agent in self.agents:
                agent.symbiosis_policy = min(1.0, agent.symbiosis_policy + 0.04)

        for agent in self.agents:
            action = agent.act("symbiotic contribution")
            turn_actions.append(action)

            if action["mercy_pass"]:
                agent.mercy_passes += 1
                agent.happiness = min(100.0, agent.happiness + 2.0)
                agent.joy = min(100.0, agent.joy + 2.5)
            else:
                agent.happiness = max(35.0, agent.happiness - 1.6)

        # Positive-sum exchange
        exchange_bonus = len([a for a in turn_actions if a["mercy_pass"]]) * (13.5 * avg_policy)
        self.heaven_metric += exchange_bonus
        self.symbiosis_index = min(6.5, self.symbiosis_index + 0.055 * avg_policy)

        mercy_compliance = sum(a["mercy_pass"] for a in turn_actions) / len(turn_actions)
        if mercy_compliance < 0.55:
            self.heaven_metric *= 0.935

        collective_reward = 1.2 if mercy_compliance > 0.72 else -0.6
        for agent in self.agents:
            agent.update_policy(collective_reward)

        self.history.append({
            "turn": turn,
            "heaven_metric": round(self.heaven_metric, 1),
            "symbiosis_index": round(self.symbiosis_index, 2),
            "mercy_compliance": round(mercy_compliance, 2),
            "predicted_heaven": round(predicted_heaven, 1),
        })

        if turn % 10 == 0 or turn == self.turns:
            print(f"Turn {turn:2d} | Heaven: {self.heaven_metric:8.1f} | Symbiosis: {self.symbiosis_index:.2f} | Mercy: {mercy_compliance:.0%} | Predicted: {predicted_heaven:7.1f}")

    def run(self):
        print("=== Reality Build Order v1 — Phase 1 + RL + World Model Simulation ===")
        print("Mapping: Group 1–3 (Foundational) + Group 4 (Symbiosis) + Group 5 (World Model lookahead)\n")

        for turn in range(1, self.turns + 1):
            self.run_turn(turn)

        print("\n=== Final Results ===")
        print(f"Final Heaven Metric: {self.heaven_metric:.1f}")
        print(f"Final Symbiosis Index: {self.symbiosis_index:.2f}")
        print(f"Total Growth: +{((self.heaven_metric - 1000) / 1000 * 100):.1f}%\n")

        if self.symbiosis_index < 3.2:
            print("Feedback: Symbiosis weak. Strengthen Group 3 (Human-AI) and Group 4 triggers.")
        elif self.heaven_metric < 12000:
            print("Feedback: Solid but slow growth. Add Group 6 (Mamba backbone) for better long-horizon efficiency.")
        else:
            print("Feedback: Strong performance. Ready for Group 7–9 (Neuromorphic + Self-Modifying + Cosmic).")


if __name__ == "__main__":
    sim = RealityBuildOrderSim(turns=50)
    sim.run()
