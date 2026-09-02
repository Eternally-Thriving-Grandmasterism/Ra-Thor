# Ra-Thor Symbiosis SDK (Python) v1.0.0

## Features
- Full 6-phase Symbiosis Handshake Protocol
- Async and sync client methods
- Robust error handling with retry logic and exponential backoff
- Custom exception hierarchy (HandshakeError, ValenceError, OntologyError, etc.)
- Valence monitoring and ontology mapping

## Async Error Handling
All async methods include:
- Automatic retry with exponential backoff (up to 3 retries)
- Specific exception types for different failure modes
- Graceful degradation on transient network failures

## Quick Start (Async with Error Handling)
```python
import asyncio
from ra_thor_symbiosis_sdk import RaThorClient
from ra_thor_symbiosis_sdk.exceptions import HandshakeError

async def main():
    client = RaThorClient("MySystem", "MyPlatform", max_retries=3)
    try:
        result = await client.start_handshake_async()
        print(result)
    except HandshakeError as e:
        print(f"Handshake failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(main())
```