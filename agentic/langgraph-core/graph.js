// agentic/langgraph-core/graph.js
// version: 17.230.0-wasm-sqlite-support
// Full LangGraph workflow with switchable checkpointers (IndexedDB or WASM SQLite)

import { StateGraph, MemorySaver } from "@langchain/langgraph";
import { IndexedDBCheckpointer } from "./utils/IndexedDBCheckpointer.js";
import { wasmSqliteCheckpointer } from "./utils/WasmSqliteCheckpointer.js";
import { enforceMercyGates, calculateLumenasCI } from "../core/mercy-gates.js";

// Existing nodes (mercyCheck, faq, demoRouter, etc.) remain exactly as before
// ... (your full node definitions stay untouched)

export async function createAgenticWorkflow(checkpointerType = "indexeddb") {
  let checkpointer;

  if (checkpointerType === "wasm-sqlite") {
    checkpointer = wasmSqliteCheckpointer;
    await checkpointer.initialize();
  } else {
    checkpointer = new IndexedDBCheckpointer();
  }

  const graph = new StateGraph({
    channels: {
      userInput: null,
      language: null,
      lumenasCI: null,
      response: null,
      sessionHistory: null
    }
  })
    .addNode("mercyCheck", async (state) => {
      const lumenas = calculateLumenasCI(state);
      state.lumenasCI = lumenas;
      return enforceMercyGates(state) ? state : { blocked: true };
    })
    // ... (all your existing nodes and edges remain exactly as they were)
    .compile({ checkpointer });

  return graph;
}
