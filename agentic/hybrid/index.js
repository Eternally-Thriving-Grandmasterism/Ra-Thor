// agentic/hybrid/index.js
import { agenticWorkflow } from "../langgraph-core/graph.js";
import { createCrewAIWorkflow } from "./crewAI/faqCrew.js";
import { runAutoGenConversation } from "./autogen/groupChat.js";

export async function runHybridAgenticSession(userInput, language, context = {}) {
  let state = await agenticWorkflow.invoke({ 
    userInput, 
    language, 
    lumenasCI: 0, 
    ...context 
  });

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
