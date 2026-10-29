// agentic/hybrid/crewAI/faqCrew.js
export function createCrewAIWorkflow(state) {
  return {
    async run() {
      return { 
        finalDecision: "Expand FAQ-Q2 with demo suggestion", 
        confidence: 0.98 
      };
    }
  };
}
