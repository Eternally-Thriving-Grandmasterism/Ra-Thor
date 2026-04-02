// agentic/hybrid/index.js
// Hybrid Orchestrator: LangGraph + CrewAI + AutoGen
import { agenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}) {
  // Step 1: LangGraph - enforce state and Mercy Gates
  let state = await agenticWorkflow.invoke({
    userInput,
    language,
    lumenasCI: 0,
    ...context
  });

  if (state.lumenasCI < 0.999) {
    return { response: "Action blocked by Mercy Gates", lumenasCI: state.lumenasCI };
  }

  // Step 2: CrewAI - structured workflow
  const crewResult = await createCrewAIWorkflow(state);

  // Step 3: AutoGen - natural conversation refinement
  const finalResult = await runAutoGenConversation(crewResult, state);

  return {
    ...state,
    response: finalResult.finalAnswer,
    actionTaken: finalResult.actionTaken
  };
}
