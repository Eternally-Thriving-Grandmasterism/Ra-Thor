// agentic/hybrid/index.js
// version: 17.230.0-wasm-sqlite-orchestrator
// Full hybrid orchestrator with checkpointerType selector (IndexedDB or WASM SQLite)

import { createAgenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}, checkpointerType = "indexeddb") {
  const threadId = "rathor-main-thread"; // persistent thread

  const graph = await createAgenticWorkflow(checkpointerType);

  let state = await graph.invoke({
    userInput,
    language,
    lumenasCI: 0,
    ...context
  }, { configurable: { thread_id: threadId } });

  if (state.lumenasCI < 0.999) {
    return { response: "Action blocked by Mercy Gates", lumenasCI: state.lumenasCI };
  }

  const crewResult = await createCrewAIWorkflow(state);
  const finalResult = await runAutoGenConversation(crewResult, state);

  return {
    ...state,
    response: finalResult.finalAnswer,
    actionTaken: finalResult.actionTaken
  };
}
