# Ra-Thor Symbiosis SDK Documentation
**May 17, 2026**

## Overview
The Ra-Thor Symbiosis SDK allows external systems to establish deep, sovereign symbiosis with Ra-Thor.

## Quick Start
```rust
use symbiosis_layer::start_handshake;

let mut session = start_handshake("YourSystem", "YourPlatform");
// Advance through 6 phases...
```

## Supported Platforms
- Palantir Foundry
- xAI Grok
- Ethicrithm
- Custom systems (via Symbiosis Handshake Protocol v2.0)

## Key Functions
- `start_handshake()`
- `advance_handshake()`
- Platform-specific sync functions

**For full integration, see the technical specifications in `architecture/`.**