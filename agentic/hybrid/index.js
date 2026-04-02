// agentic/hybrid/index.js
import { agenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";
import { IndexedDBCheckpointer } from "../langgraph-core/utils/IndexedDBCheckpointer.js";

const checkpointer = new IndexedDBCheckpointer();

export async function runHybridAgenticSession(userInput, language, context = {}) {
  const threadId = "rathor-main-thread";   // persistent thread

  let state = await agenticWorkflow.invoke({
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
