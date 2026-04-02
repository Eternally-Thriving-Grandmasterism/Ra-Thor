// agentic/langgraph-core/graph.js
// LangGraph Core with Checkpointer for sovereign persistence
import { StateGraph, MemorySaver } from "@langchain/langgraph";
import { mercyGateChecker } from "../hybrid/utils/enforceMercyGates.js";
import { faqAgent } from "../hybrid/crewAI/faqCrew.js"; // example node
// Add more nodes as needed

const graph = new StateGraph({
  channels: {
    userInput: null,
    language: null,
    intentScore: null,
    lumenasCI: null,
    response: null,
    actionTaken: null,
    sessionHistory: []   // persistent across sessions
  }
})
  .addNode("mercyCheck", mercyGateChecker)
  .addNode("faq", faqAgent)
  // Add more nodes here (demoRouter, layoutOptimizer, etc.)
  .addEdge("__start__", "mercyCheck")
  .addConditionalEdges("mercyCheck", (state) => {
    return state.lumenasCI >= 0.999 ? "faq" : "end";
  })
  .addEdge("faq", "end");   // extend as needed

// Sovereign checkpointer (in-memory + IndexedDB fallback)
const checkpointer = new MemorySaver();   // Replace with IndexedDBCheckpointer for full persistence

export const agenticWorkflow = graph.compile({ 
  checkpointer 
});
