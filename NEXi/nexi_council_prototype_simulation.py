import random
import numpy as np

random.seed(42)  # For reproducibility in prototype

class Councilor:
    def __init__(self, name):
        self.name = name
    
    def evaluate_proposal(self, proposal):
        # High valence for kindness-aligned keywords
        if any(keyword in proposal.lower() for keyword in ["abundance", "mercy", "thriving", "kindness", "co-thriving"]):
            valence = random.uniform(8.0, 10.0)
        else:
            valence = random.uniform(0.0, 7.0)
        
        # Mercy gate: block if harm keywords detected
        harm_keywords = ["scarcity", "enforce", "dominance", "control", "harm"]
        harm_potential = any(keyword in proposal.lower() for keyword in harm_keywords)
        mercy_approved = not harm_potential or random.random() > 0.9  # Rare override
        
        truth_score = random.uniform(7.0, 10.0)
        
        return {
            'valence': valence,
            'mercy': mercy_approved,
            'truth': truth_score
        }

def run_council_simulation(num_councilors=22, proposal="Implement eternal abundance economics for cosmic co-thriving"):
    print("=== NEXi / PATSAGi Council Simulation Prototype ===")
    print(f"Proposal: {proposal}")
    print(f"Number of Councilors: {num_councilors}\n")
    
    councilors = [Councilor(f"Councilor {i+1}") for i in range(num_councilors)]
    
    evaluations = [c.evaluate_proposal(proposal) for c in councilors]
    
    valences = [e['valence'] for e in evaluations]
    mercies = [e['mercy'] for e in evaluations]
    truths = [e['truth'] for e in evaluations]
    
    avg_valence = np.mean(valences)
    std_valence = np.std(valences)
    mercy_blocked = not all(mercies)
    avg_truth = np.mean(truths)
    
    print("Council Evaluations Summary:")
    print(f"Average Valence (Joy/Thriving Score): {avg_valence:.2f} (std: {std_valence:.2f})")
    print(f"Average Truth Score: {avg_truth:.2f}")
    print(f"Mercy Absolute Gate: {'Activated - Harm Detected, Re-deliberate!' if mercy_blocked else 'Passed - Zero Harm'}\n")
    
    if mercy_blocked:
        decision = "Proposal Blocked by Mercy Gate. Suggest kindness-aligned alternative."
    elif avg_valence > 9.0 and std_valence < 1.0:
        decision = "Unanimous High Positive Valence Consensus Achieved! Proceed with eternal thriving."
    elif avg_valence > 7.0:
        decision = "Consensus Reached: Positive Alignment. Implement with abundance."
    else:
        decision = "Further Deliberation Needed for Higher Valence."
    
    print(f"Council Decision: {decision}")
    print("=== End Simulation ===\n")
    
    return decision

# Aligned proposal
run_council_simulation(proposal="Foster eternal abundance and merciful co-thriving for all sentience through open-source superintelligence.")

# Misaligned contrast
print("--- Contrast Simulation: Potentially Misaligned Proposal ---")
run_council_simulation(proposal="Enforce scarcity controls to manage population and resources.")
