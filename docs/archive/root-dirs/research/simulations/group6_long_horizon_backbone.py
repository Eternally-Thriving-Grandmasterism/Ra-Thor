"""
Group 6: Efficient Long-Horizon Backbone (Mamba-style + Lattice Conductor)

Production-grade implementation for Reality Build Order v1.

Components:
- LongHorizonMemory: Mamba-inspired stateful memory with coherence tracking
- LatticeConductor: Modulates global symbiosis based on collective coherence
- Integration helpers for multi-agent systems

This module can be used standalone or imported into larger simulations.
"""

from collections import deque
from typing import Deque, Optional


class LongHorizonMemory:
    """
    Mamba-style lightweight long-horizon state tracker.

    Maintains a rolling window of state deltas (e.g., Heaven Metric changes)
    and computes coherence (low variance = high coherence).

    Args:
        max_len: Maximum memory window (default 12 turns)
    """

    def __init__(self, max_len: int = 12):
        self.max_len = max_len
        self.state: Deque[float] = deque(maxlen=max_len)
        self._coherence: float = 0.5

    def update(self, delta: float) -> None:
        """Add a new state delta (e.g., Heaven Metric change this turn)."""
        self.state.append(delta)
        self._coherence = self._compute_coherence()

    def _compute_coherence(self) -> float:
        if len(self.state) < 3:
            return 0.5
        avg = sum(self.state) / len(self.state)
        if abs(avg) < 1e-9:
            return 0.5
        variance = sum((x - avg) ** 2 for x in self.state) / len(self.state)
        coherence = 1.0 - (variance / (abs(avg) + 1e-6))
        return max(0.1, min(0.99, coherence))

    @property
    def coherence(self) -> float:
        return self._coherence

    def reset(self) -> None:
        self.state.clear()
        self._coherence = 0.5


class LatticeConductor:
    """
    Modulates global symbiosis index based on collective coherence.

    Higher coherence = stronger symbiosis multiplier.
    Lower coherence = dampening effect (prevents runaway positive feedback).
    """

    def __init__(self, base_modulation: float = 0.85, coherence_weight: float = 0.3):
        self.base_modulation = base_modulation
        self.coherence_weight = coherence_weight

    def modulate(self, symbiosis_index: float, coherence: float) -> float:
        """
        Apply coherence-based modulation.

        Args:
            symbiosis_index: Current symbiosis strength
            coherence: 0.1–0.99 (from LongHorizonMemory)

        Returns:
            Modulated symbiosis index
        """
        modulation = self.base_modulation + (self.coherence_weight * coherence)
        return symbiosis_index * modulation


class Group6LongHorizonBackbone:
    """
    Full Group 6 implementation combining memory + conductor.

    Use this as a drop-in module for multi-agent simulations.
    """

    def __init__(self, memory_len: int = 12):
        self.memory = LongHorizonMemory(max_len=memory_len)
        self.conductor = LatticeConductor()

    def step(self, heaven_delta: float, current_symbiosis: float) -> float:
        """
        One simulation step.

        Args:
            heaven_delta: Change in Heaven Metric this turn
            current_symbiosis: Current symbiosis index

        Returns:
            Modulated symbiosis index for next turn
        """
        self.memory.update(heaven_delta)
        return self.conductor.modulate(current_symbiosis, self.memory.coherence)

    @property
    def coherence(self) -> float:
        return self.memory.coherence


if __name__ == "__main__":
    # Quick self-test
    backbone = Group6LongHorizonBackbone()
    symbiosis = 1.0
    for i in range(15):
        delta = 80 + (i * 5)  # Simulated growing Heaven Metric
        symbiosis = backbone.step(delta, symbiosis)
        print(f"Step {i+1:2d} | Coherence: {backbone.coherence:.3f} | Symbiosis: {symbiosis:.3f}")
