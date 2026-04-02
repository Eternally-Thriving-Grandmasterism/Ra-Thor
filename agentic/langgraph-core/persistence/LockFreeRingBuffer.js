// agentic/langgraph-core/persistence/LockFreeRingBuffer.js
// version: 17.245.0-lock-free-ring-buffer
// Lock-Free Ring Buffer using SharedArrayBuffer + Atomics
// Zero-contention, high-throughput task queue for WorkerPool
// Fully Mercy-Gated and sovereign

export class LockFreeRingBuffer {
  constructor(capacity = 1024) {
    // Header (8 bytes) + payload buffer
    this.buffer = new SharedArrayBuffer(8 + capacity * 4); // 4 bytes per task ID (Uint32)
    this.capacity = capacity;
    
    // Atomics views
    this.head = new Int32Array(this.buffer, 0, 1);   // write pointer
    this.tail = new Int32Array(this.buffer, 4, 1);   // read pointer
    this.data = new Uint32Array(this.buffer, 8);     // actual ring buffer
    
    // Initialize
    Atomics.store(this.head, 0, 0);
    Atomics.store(this.tail, 0, 0);
  }

  // Enqueue a task ID (or any 32-bit value)
  enqueue(taskId) {
    const head = Atomics.load(this.head, 0);
    const nextHead = (head + 1) % this.capacity;
    const tail = Atomics.load(this.tail, 0);

    if (nextHead === tail) {
      return false; // full
    }

    this.data[head] = taskId;
    Atomics.store(this.head, 0, nextHead);
    Atomics.notify(this.head, 0); // wake any waiting readers
    return true;
  }

  // Dequeue a task ID (returns null if empty)
  dequeue() {
    const tail = Atomics.load(this.tail, 0);
    const head = Atomics.load(this.head, 0);

    if (tail === head) {
      return null; // empty
    }

    const taskId = this.data[tail];
    const nextTail = (tail + 1) % this.capacity;
    Atomics.store(this.tail, 0, nextTail);
    Atomics.notify(this.tail, 0);
    return taskId;
  }

  // Non-blocking peek (for diagnostics)
  peek() {
    const tail = Atomics.load(this.tail, 0);
    const head = Atomics.load(this.head, 0);
    if (tail === head) return null;
    return this.data[tail];
  }

  // Current size
  size() {
    const head = Atomics.load(this.head, 0);
    const tail = Atomics.load(this.tail, 0);
    return (head - tail + this.capacity) % this.capacity;
  }

  // Reset buffer (for testing / recovery)
  reset() {
    Atomics.store(this.head, 0, 0);
    Atomics.store(this.tail, 0, 0);
  }
}

// Singleton for global use in WorkerPool
export const lockFreeRingBuffer = new LockFreeRingBuffer(2048); // 2K capacity by default
