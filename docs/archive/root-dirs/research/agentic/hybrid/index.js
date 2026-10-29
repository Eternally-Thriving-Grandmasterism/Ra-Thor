// agentic/hybrid/index.js
// version: 17.242.0-langgraph-persistence-layer
// Full hybrid orchestrator using the new unified Persistence Layer

import { createAgenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}, preferredCheckpointer = "indexeddb") {
  const threadId = "rathor-main-thread";

  const graph = await createAgenticWorkflow(preferredCheckpointer);

  let state = await graph.invoke({
    userInput,
    language,
    lumenasCI: 0,
    ...context
  }, { configurable: { thread_id: threadId } });

  if (state.lumenasCI < 0.999) {
    return { response: "Action blocked by Mercy Gates", lumenasCI: state.lumenasCI, checkpointerUsed: preferredCheckpointer };
  }

  const crewResult = await createCrewAIWorkflow(state);
  const finalResult = await runAutoGenConversation(crewResult, state);

  return {
    ...state,
    response: finalResult.finalAnswer,
    actionTaken: finalResult.actionTaken,
    checkpointerUsed: preferredCheckpointer
  };
}
