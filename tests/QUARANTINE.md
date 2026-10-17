e2e.test.js — imports ../core/ra-thor-sovereign-orchestrator.js, which is not shipped at repo root; exercises WASM/WebLLM/browser flows.
integration.test.js — imports ../core/ra-thor-sovereign-orchestrator.js, which is not shipped at repo root; exercises WASM/WebLLM/browser flows.
memory-leak.test.js — imports ../core/ra-thor-sovereign-orchestrator.js, which is not shipped at repo root; uses Chrome-only performance.memory and is not valid in plain Node CI.
orchestrator.test.js — imports ../core/ra-thor-sovereign-orchestrator.js, which is not shipped at repo root; exercises WASM/WebLLM/browser flows.
security-vulnerability.test.js — imports ../core/ra-thor-sovereign-orchestrator.js, which is not shipped at repo root; exercises WASM/WebLLM/browser flows.
