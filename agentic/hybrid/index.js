// agentic/hybrid/index.js
// version: 17.235.0-absurd-sql-support
// Full hybrid orchestrator with absurd-sql VFS as a first-class checkpointerType
// Graceful fallback chain: absurd-sql → wasm-sqlite → indexeddb

import { createAgenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";
import { absurdSqlCheckpointer } from "../langgraph-core/utils/AbsurdSqlCheckpointer.js";

export async function runHybridAgenticSession(userInput, language, context = {}, preferredCheckpointer = "indexeddb") {
  const threadId = "rathor-main-thread";

  let graph;
  let checkpointerType = preferredCheckpointer;

  // Intelligent fallback chain
  try {
    if (preferredCheckpointer === "absurd-sql") {
      graph = await createAgenticWorkflow("absurd-sql");
    } else if (preferredCheckpointer === "wasm-sqlite") {
      graph = await createAgenticWorkflow("wasm-sqlite");
    } else {
      graph = await createAgenticWorkflow("indexeddb");
    }
  } catch (error) {
    console.warn(`Preferred checkpointer ${preferredCheckpointer} failed — falling back to IndexedDB:`, error.message);
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
