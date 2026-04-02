# SharedArrayBuffer Concurrency Patterns for Rathor.ai (2026)

Practical, production-ready concurrency patterns using SharedArrayBuffer + Atomics for our Worker Pool and VFS checkpointers. All patterns are sovereign, offline-first, and Mercy-Gated.

## 1. Atomics Signaling Pattern (Basic Coordination)

**Use case**: Worker → Main thread signaling (e.g., “checkpoint saved”).

```javascript
// In Worker
const signal = new Int32Array(sharedBuffer, 0, 1); // offset 0
Atomics.store(signal, 0, 1);
Atomics.notify(signal, 0);

// In Main thread
const signal = new Int32Array(sharedBuffer, 0, 1);
Atomics.wait(signal, 0, 0, 5000); // wait up to 5s
