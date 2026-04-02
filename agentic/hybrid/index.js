// agentic/hybrid/index.js
// version: 17.231.0-graceful-wasm-fallback
// Full hybrid orchestrator with intelligent checkpointer fallback
// WASM SQLite preferred, falls back to IndexedDB seamlessly

import { createAgenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}, preferredCheckpointer = "wasm-sqlite") {
  const threadId = "rathor-main-thread";

  let graph;
  let checkpointerType = preferredCheckpointer;

  // Graceful fallback logic
  try {
    if (preferredCheckpointer === "wasm-sqlite") {
      graph = await createAgenticWorkflow("wasm-sqlite");
    } else {
      graph = await createAgenticWorkflow("indexeddb");
    }
  } catch (error) {
    console.warn("WASM SQLite failed to initialize — falling back to IndexedDB:", error.message);
    checkpointerType = "indexeddb";
    graph = await createAgenticWorkflow("indexeddb");
  }

  let state = await graph.invoke({
    userInput,
    language,
    lumenasCI: 0,
    ...context
  }, { configurable: { thread_id: threadId } });

  if (state.lumenasCI < 0.999) {
    return { response: "Action blocked by Mercy Gates", lumenasCI: state.lumenasCI, checkpointerUsed: checkpointerType };
  }

  const crewResult = await createCrewAIWorkflow(state);
  const finalResult = await runAutoGenConversation(crewResult, state);

  return {
    ...state,
    response: finalResult.finalAnswer,
    actionTaken: finalResult.actionTaken,
    checkpointerUsed: checkpointerType
  };
}
