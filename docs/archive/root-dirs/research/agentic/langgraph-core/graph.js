// agentic/langgraph-core/graph.js
// version: 17.242.0-langgraph-persistence-layer
// Full LangGraph workflow using the new unified Persistence Layer

import { StateGraph } from "@langchain/langgraph";
import { langGraphPersistence } from "./persistence/LangGraphPersistenceLayer.js";
import { enforceMercyGates, calculateLumenasCI } from "../core/mercy-gates.js";

export async function createAgenticWorkflow(checkpointerType = "indexeddb") {
  // Force the persistence layer to use the requested type
  langGraphPersistence.checkpointer.preferredType = checkpointerType;
  await langGraphPersistence.initialize();

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
    // ... (all existing nodes and edges remain exactly as before)
    .compile({ checkpointer: langGraphPersistence });

  return graph;
}
