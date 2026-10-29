// agentic/langgraph-core/persistence/LangGraphPersistenceLayer.js
// version: 17.242.0-langgraph-persistence-layer
// Unified high-level Persistence Layer for LangGraph
// Wraps VFSCheckpointer, adds thread management, state validation, Mercy Gates, profiling

import { vfsCheckpointer } from "../utils/VFSCheckpointer.js";
import { enforceMercyGates, calculateLumenasCI } from "../../core/mercy-gates.js";

export class LangGraphPersistenceLayer {
  constructor(preferredType = "indexeddb") {
    this.checkpointer = vfsCheckpointer;
    this.checkpointer.preferredType = preferredType;
    this.threads = new Map(); // threadId → lastCheckpoint
  }

  async initialize() {
    return await this.checkpointer.initialize();
  }

  async save(state, config) {
    const threadId = config?.configurable?.thread_id || "rathor-main-thread";
    const lumenas = calculateLumenasCI(state);

    if (lumenas < 0.999) {
      console.warn(`Mercy Gate blocked persistence for thread ${threadId}`);
      return { blocked: true };
    }

    const start = performance.now();
    const success = await this.checkpointer.save(state, threadId);
    const duration = (performance.now() - start).toFixed(2);

    if (success) {
      this.threads.set(threadId, { timestamp: Date.now(), lumenas });
      console.log(`✅ Persisted thread ${threadId} in ${duration} ms (lumenas: ${lumenas.toFixed(3)})`);
    }
    return { success, durationMs: duration, lumenas };
  }

  async load(config) {
    const threadId = config?.configurable?.thread_id || "rathor-main-thread";
    const start = performance.now();
    const state = await this.checkpointer.load(threadId);
    const duration = (performance.now() - start).toFixed(2);

    if (state) {
      this.threads.set(threadId, { timestamp: Date.now(), lumenas: state.lumenasCI || 0 });
    }
    return state;
  }

  close() {
    this.checkpointer.close();
    this.threads.clear();
  }

  getThreadStats(threadId) {
    return this.threads.get(threadId) || null;
  }
}

// Singleton for global LangGraph use
export const langGraphPersistence = new LangGraphPersistenceLayer();
