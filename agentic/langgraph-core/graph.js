// agentic/langgraph-core/graph.js
// version: 17.230.0-wasm-sqlite-support

import { StateGraph, MemorySaver } from "@langchain/langgraph";
import { IndexedDBCheckpointer } from "./utils/IndexedDBCheckpointer.js";
import { wasmSqliteCheckpointer } from "./utils/WasmSqliteCheckpointer.js";
import { enforceMercyGates } from "../core/mercy-gates.js";

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
    // ... (rest of your existing nodes remain unchanged)
    .compile({ checkpointer });

  return graph;
}
