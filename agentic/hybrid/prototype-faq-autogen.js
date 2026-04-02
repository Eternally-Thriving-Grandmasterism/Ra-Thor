// Minimal working prototype
async function testPrototype(userInput) {
  console.log("🚀 Starting hybrid prototype...");
  const state = { userInput, language: "en", lumenasCI: 1.0 };

  const crewResult = { finalDecision: "Expand FAQ-Q2" };
  const finalAnswer = "Predicted answer for Q2 + personalized demo link";

  console.log("✅ Prototype complete:", finalAnswer);
  return finalAnswer;
}

testPrototype("commercial licensing");
