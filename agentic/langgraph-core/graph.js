// agentic/langgraph-core/graph.js
// version: 17.241.0-vfs-abstraction
// Full LangGraph workflow using the new unified VFSCheckpointer

import { StateGraph } from "@langchain/langgraph";
import { vfsCheckpointer } from "./utils/VFSCheckpointer.js";
import { enforceMercyGates, calculateLumenasCI } from "../core/mercy-gates.js";

export async function createAgenticWorkflow(checkpointerType = "indexeddb") {
  // Unified VFS abstraction handles everything
  const checkpointer = vfsCheckpointer; // singleton
  // Force preferred type if different from default
  if (checkpointer.preferredType !== checkpointerType) {
    checkpointer.preferredType = checkpointerType;
  }

  await checkpointer.initialize();

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
