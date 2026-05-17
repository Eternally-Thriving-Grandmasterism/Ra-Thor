"""Bulkhead Pattern Implementation for Ra-Thor SDK

The Bulkhead Pattern isolates different components so that failures in one
area don't cascade to others (like watertight compartments in a ship).
"""

import asyncio
from typing import Callable, Any, Dict
from dataclasses import dataclass

@dataclass
class BulkheadConfig:
    max_concurrent: int = 10          # Maximum concurrent operations
    queue_size: int = 100             # Maximum queued requests
    timeout_seconds: float = 30.0     # Timeout for operations

class Bulkhead:
    def __init__(self, name: str, config: BulkheadConfig = None):
        self.name = name
        self.config = config or BulkheadConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.active_count = 0

    async def execute(self, coro: Callable, *args, **kwargs) -> Any:
        """Execute a coroutine within the bulkhead limits"""
        async with self.semaphore:
            self.active_count += 1
            try:
                return await asyncio.wait_for(
                    coro(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            finally:
                self.active_count -= 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": self.active_count,
            "max_concurrent": self.config.max_concurrent,
            "queue_size": self.config.queue_size,
        }

# Pre-configured bulkheads for different Ra-Thor components
PALANTIR_BULKHEAD = Bulkhead("palantir", BulkheadConfig(max_concurrent=5))
XAI_BULKHEAD = Bulkhead("xai", BulkheadConfig(max_concurrent=8))
ETHICRITHM_BULKHEAD = Bulkhead("ethicrithm", BulkheadConfig(max_concurrent=3))
QUANTUM_BULKHEAD = Bulkhead("quantum", BulkheadConfig(max_concurrent=20))