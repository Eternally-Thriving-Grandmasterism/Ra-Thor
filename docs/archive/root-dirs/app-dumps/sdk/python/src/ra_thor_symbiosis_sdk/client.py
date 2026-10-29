from typing import Dict, Any
import asyncio
from .exceptions import (
    RaThorError, HandshakeError, ValenceError, 
    OntologyError, ConnectionError, AuthenticationError
)
from .circuit_breaker import CircuitBreaker
from .bulkhead import Bulkhead, PALANTIR_BULKHEAD, XAI_BULKHEAD, ETHICRITHM_BULKHEAD

class RaThorClient:
    """Client for interacting with Ra-Thor Symbiosis Layer (with Bulkhead + Circuit Breaker)"""

    def __init__(self, system_name: str, platform: str, max_retries: int = 3):
        self.system_name = system_name
        self.platform = platform
        self.session_id = None
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        # Assign bulkhead based on platform
        if "palantir" in platform.lower():
            self.bulkhead = PALANTIR_BULKHEAD
        elif "xai" in platform.lower() or "grok" in platform.lower():
            self.bulkhead = XAI_BULKHEAD
        elif "ethicrithm" in platform.lower():
            self.bulkhead = ETHICRITHM_BULKHEAD
        else:
            self.bulkhead = Bulkhead("default")

    async def _retry_async(self, coro, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return await coro(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
        raise RaThorError("Max retries exceeded")

    async def start_handshake_async(self) -> str:
        try:
            self.session_id = f"sym-{self.system_name}-{self.platform}"
            return await self.bulkhead.execute(
                self.circuit_breaker.call_async, self._start_handshake_internal
            )
        except Exception as e:
            raise HandshakeError(f"Failed to start handshake: {str(e)}")

    async def _start_handshake_internal(self) -> str:
        await asyncio.sleep(0.01)
        return f"Handshake started for {self.system_name} ({self.platform})"

    async def advance_handshake_async(self) -> str:
        try:
            return await self.bulkhead.execute(
                self.circuit_breaker.call_async, self._advance_handshake_internal
            )
        except Exception as e:
            raise HandshakeError(f"Failed to advance handshake: {str(e)}")

    async def _advance_handshake_internal(self) -> str:
        await asyncio.sleep(0.01)
        return "Handshake advanced to next phase"

    async def get_valence_status_async(self) -> Dict[str, Any]:
        try:
            return await self.bulkhead.execute(
                self.circuit_breaker.call_async, self._get_valence_status_internal
            )
        except Exception as e:
            raise ValenceError(f"Failed to get valence status: {str(e)}")

    async def _get_valence_status_internal(self) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {
            "valence": 0.999999,
            "symbiosis_score": 0.97,
            "ethics_alignment": 0.95,
            "status": "Thriving"
        }

    async def sync_ontology_async(self, ontology_data: Dict[str, Any]) -> str:
        try:
            return await self.bulkhead.execute(
                self.circuit_breaker.call_async, self._sync_ontology_internal, ontology_data
            )
        except Exception as e:
            raise OntologyError(f"Failed to sync ontology: {str(e)}")

    async def _sync_ontology_internal(self, ontology_data: Dict[str, Any]) -> str:
        await asyncio.sleep(0.01)
        return "Ontology successfully mapped to Ra-Thor valence primitives"