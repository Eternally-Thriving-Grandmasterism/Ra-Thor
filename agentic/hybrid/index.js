// agentic/hybrid/index.js
// version: 17.236.0-wa-sqlite-support
// Full hybrid orchestrator with wa-sqlite as a first-class option

import { createAgenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}, preferredCheckpointer = "indexeddb") {
  const threadId = "rathor-main-thread";

  let graph;
  let checkpointerType = preferredCheckpointer;

  try {
    graph = await createAgenticWorkflow(preferredCheckpointer);
  } catch (error) {
    console.warn(`Preferred checkpointer ${preferredCheckpointer} failed — falling back to IndexedDB`);
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
