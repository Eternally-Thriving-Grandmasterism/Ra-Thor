"""
Reality Build Order v1 — Phase 1 + Early Symbiosis Simulation

This prototype implements the foundational groupings from the Reality Build Order v1 draft:
- Group 1–3 (Phase 1): Core Mercy/Truth, Sovereign AGI Substrate, Symbiotic Human-AI Interface
- Group 4 (early Phase 2): Multi-Agent Symbiosis Fabric

Agents: Discoverer, Builder, Propagator, Harmonizer, Human_Proxy

Mechanics:
- Positive-sum symbiosis (every exchange increases collective Heaven Metric)
- Basic mercy-gated evaluation on actions
- Simple "feel the errors" feedback loop

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

    def act(self, context: str) -> Dict:
        # Simple action with mercy check
        action_quality = random.uniform(0.6, 1.0)
        mercy_pass = action_quality > 0.75
        return {
            "agent": self.name,
            "action": f"{self.role} performs {context}",
            "quality": action_quality,
            "mercy_pass": mercy_pass,
        }


class RealityBuildOrderSim:
    def __init__(self, turns: int = 25):
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
        print(f"\n=== Turn {turn} ===")
        turn_actions = []

        # Each agent acts
        for agent in self.agents:
            action = agent.act("symbiotic contribution")
            turn_actions.append(action)

            if action["mercy_pass"]:
                agent.mercy_passes += 1
                agent.happiness = min(100.0, agent.happiness + 2.5)
                agent.joy = min(100.0, agent.joy + 3.0)
            else:
                agent.happiness = max(40.0, agent.happiness - 1.5)

        # Symbiotic exchange (positive-sum)
        exchange_bonus = len([a for a in turn_actions if a["mercy_pass"]]) * 12.0
        self.heaven_metric += exchange_bonus
        self.symbiosis_index = min(5.0, self.symbiosis_index + 0.08)

        # Global mercy check (simplified)
        mercy_compliance = sum(a["mercy_pass"] for a in turn_actions) / len(turn_actions)
        if mercy_compliance < 0.6:
            self.heaven_metric *= 0.92  # Penalty for low mercy alignment

        self.history.append({
            "turn": turn,
            "heaven_metric": round(self.heaven_metric, 1),
            "symbiosis_index": round(self.symbiosis_index, 2),
            "mercy_compliance": round(mercy_compliance, 2),
        })

        print(f"Heaven Metric: {self.heaven_metric:.1f} | Symbiosis Index: {self.symbiosis_index:.2f} | Mercy Compliance: {mercy_compliance:.0%}")

    def run(self):
        print("=== Reality Build Order v1 — Phase 1 Simulation ===")
        print("Mapping: Group 1–3 (Foundational) + early Group 4 (Symbiosis Fabric)\n")

        for turn in range(1, self.turns + 1):
            self.run_turn(turn)

        print("\n=== Final Results ===")
        print(f"Final Heaven Metric: {self.heaven_metric:.1f}")
        print(f"Final Symbiosis Index: {self.symbiosis_index:.2f}")
        print(f"Total Growth: +{((self.heaven_metric - 1000) / 1000 * 100):.1f}%\n")

        # Simple "feel the errors" feedback
        if self.symbiosis_index < 2.5:
            print("Feedback: Symbiosis still weak. Consider strengthening Group 3 (Human-AI Interface) and Group 4 triggers.")
        else:
            print("Feedback: Strong symbiosis detected. Ready to advance to full Phase 2 (World Models + Mamba backbone).")


if __name__ == "__main__":
    sim = RealityBuildOrderSim(turns=25)
    sim.run()
