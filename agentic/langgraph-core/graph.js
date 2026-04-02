// agentic/langgraph-core/graph.js
// version: 17.236.0-wa-sqlite-support
// Full LangGraph workflow with support for wa-sqlite as checkpointerType

import { StateGraph } from "@langchain/langgraph";
import { IndexedDBCheckpointer } from "./utils/IndexedDBCheckpointer.js";
import { wasmSqliteCheckpointer } from "./utils/WasmSqliteCheckpointer.js";
import { absurdSqlCheckpointer } from "./utils/AbsurdSqlCheckpointer.js";
import { waSqliteCheckpointer } from "./utils/WaSqliteCheckpointer.js";
import { enforceMercyGates, calculateLumenasCI } from "../core/mercy-gates.js";

export async function createAgenticWorkflow(checkpointerType = "indexeddb") {
  let checkpointer;

  switch (checkpointerType) {
    case "wa-sqlite":
      checkpointer = waSqliteCheckpointer;
      await checkpointer.initialize();
      break;
    case "wasm-sqlite":
      checkpointer = wasmSqliteCheckpointer;
      await checkpointer.initialize();
      break;
    case "absurd-sql":
      checkpointer = absurdSqlCheckpointer;
      await checkpointer.initialize();
      break;
    default:
      checkpointer = new IndexedDBCheckpointer();
  }

  const graph = new StateGraph({
    channels: { /* your existing channels */ }
  })
    .addNode("mercyCheck", async (state) => {
      const lumenas = calculateLumenasCI(state);
      state.lumenasCI = lumenas;
      return enforceMercyGates(state) ? state : { blocked: true };
    })
    // ... (all existing nodes remain unchanged)
    .compile({ checkpointer });

  return graph;
}
