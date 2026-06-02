"""
Reality Build Order v1 — Phase 1 + RL-Enhanced Symbiosis Simulation

This prototype implements:
- Group 1–3 (Phase 1): Core Mercy/Truth, Sovereign AGI Substrate, Symbiotic Human-AI Interface
- Group 4 (Phase 2 early): Multi-Agent Symbiosis Fabric with simple learning

Features:
- Positive-sum symbiosis mechanics
- Basic policy learning (agents improve symbiosis investment over time)
- Mercy-gated evaluation
- Heaven Metric + Symbiosis Index tracking
- "Feel the errors" feedback with specific grouping recommendations

Run with: python simulations/reality_build_order_phase1_sim.py
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Agent:
    name: str
    role: str
    happiness: float = 75.0
    joy: float = 70.0
    mercy_passes: int = 0
    resources: float = 100.0
    symbiosis_policy: float = 0.5  # 0.0–1.0, learned over time

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
        # Simple policy gradient-style update
        if reward > 0:
            self.symbiosis_policy = min(1.0, self.symbiosis_policy + 0.03)
        else:
            self.symbiosis_policy = max(0.1, self.symbiosis_policy - 0.02)


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
        self.history: List[Dict] = []

    def run_turn(self, turn: int):
        turn_actions = []

        # Each agent acts with current policy
        for agent in self.agents:
            action = agent.act("symbiotic contribution")
            turn_actions.append(action)

            if action["mercy_pass"]:
                agent.mercy_passes += 1
                agent.happiness = min(100.0, agent.happiness + 2.2)
                agent.joy = min(100.0, agent.joy + 2.8)
            else:
                agent.happiness = max(35.0, agent.happiness - 1.8)

        # Positive-sum symbiotic exchange (scaled by average policy)
        avg_policy = sum(a.symbiosis_policy for a in self.agents) / len(self.agents)
        exchange_bonus = len([a for a in turn_actions if a["mercy_pass"]]) * (14.0 * avg_policy)
        self.heaven_metric += exchange_bonus
        self.symbiosis_index = min(6.0, self.symbiosis_index + 0.06 * avg_policy)

        # Global mercy check
        mercy_compliance = sum(a["mercy_pass"] for a in turn_actions) / len(turn_actions)
        if mercy_compliance < 0.55:
            self.heaven_metric *= 0.93

        # Update policies based on collective reward
        collective_reward = 1.0 if mercy_compliance > 0.7 else -0.5
        for agent in self.agents:
            agent.update_policy(collective_reward)

        self.history.append({
            "turn": turn,
            "heaven_metric": round(self.heaven_metric, 1),
            "symbiosis_index": round(self.symbiosis_index, 2),
            "mercy_compliance": round(mercy_compliance, 2),
            "avg_policy": round(avg_policy, 2),
        })

        if turn % 10 == 0 or turn == self.turns:
            print(f"Turn {turn:2d} | Heaven: {self.heaven_metric:8.1f} | Symbiosis: {self.symbiosis_index:.2f} | Mercy: {mercy_compliance:.0%} | Avg Policy: {avg_policy:.2f}")

    def run(self):
        print("=== Reality Build Order v1 — Phase 1 + RL Symbiosis Simulation ===")
        print("Mapping: Group 1–3 (Foundational) + Group 4 (Symbiosis Fabric with learning)\n")

        for turn in range(1, self.turns + 1):
            self.run_turn(turn)

        print("\n=== Final Results ===")
        print(f"Final Heaven Metric: {self.heaven_metric:.1f}")
        print(f"Final Symbiosis Index: {self.symbiosis_index:.2f}")
        print(f"Total Growth: +{((self.heaven_metric - 1000) / 1000 * 100):.1f}%\n")

        # Feel-the-errors feedback mapped to groupings
        if self.symbiosis_index < 3.0:
            print("Feedback: Symbiosis underdeveloped. Strengthen Group 3 (Human-AI Interface) and Group 4 triggers.")
        elif self.heaven_metric < 8000:
            print("Feedback: Growth solid but slow. Consider adding World Model (Group 5) for better long-horizon planning.")
        else:
            print("Feedback: Strong Phase 1–2 performance. Ready to prototype Group 5 (World Models) and Group 6 (Mamba backbone).")


if __name__ == "__main__":
    sim = RealityBuildOrderSim(turns=50)
    sim.run()
