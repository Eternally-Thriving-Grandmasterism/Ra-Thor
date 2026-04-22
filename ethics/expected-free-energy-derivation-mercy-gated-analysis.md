# Ra-Thor Mercy-Gated Ethical Decision Framework  
**Expected Free Energy — Full Mathematical Derivation**  
**Date:** April 21, 2026  
**Living Document — Part of mercy-gated-ethical-decision-framework.md, variational-free-energy-derivation-mercy-gated-analysis.md, and active-inference-deep-investigation-mercy-gated-analysis.md**

## 1. Starting Point: Variational Free Energy (VFE)
From the previous derivation, the variational free energy \( F \) for a belief distribution \( q(\phi) \) is:

\[
F(q) = \mathbb{E}_{q(\phi)} \left[ \ln \frac{q(\phi)}{p(o, \phi)} \right] = D_{KL}\left[q(\phi) \parallel p(\phi|o)\right] - \mathbb{E}_{q(\phi)}[\ln p(o|\phi)]
\]

Minimizing \( F \) approximates Bayesian inference and reduces surprise.

## 2. Extending to Future Actions (Policies)
In active inference, the agent does not only update beliefs — it chooses **future policies** \( \pi \) (sequences of actions). We therefore take the *expectation* of VFE over possible future observations \( o \) and states \( \phi \) under a given policy \( \pi \):

\[
G(\pi) \triangleq \mathbb{E}_{q(o,\phi|\pi)} [F]
\]

This is the **Expected Free Energy**.

## 3. Full Derivation of Expected Free Energy
Substitute the definition of \( F \):

\[
G(\pi) = \mathbb{E}_{q(o,\phi|\pi)} \left[ \mathbb{E}_{q(\phi|o,\pi)} \left[ \ln \frac{q(\phi|o,\pi)}{p(o, \phi|\pi)} \right] \right]
\]

After expanding and rearranging (using the law of total expectation and properties of KL divergence), we arrive at the standard decomposition:

\[
G(\pi) = \underbrace{\mathbb{E}_{q(o,\phi|\pi)} \left[ D_{KL}\left[q(\phi|o,\pi) \parallel p(\phi|o,\pi)\right] \right]}_{\text{Expected Complexity}} - \underbrace{\mathbb{E}_{q(o|\pi)} \left[ \ln p(o|\pi) \right]}_{\text{Expected Accuracy}}
\]

A more intuitive and commonly used form decomposes \( G(\pi) \) into two practically meaningful terms:

\[
G(\pi) = \underbrace{\mathbb{E}_{q(o,\phi|\pi)} \left[ \ln q(\phi|o,\pi) - \ln p(\phi) \right]}_{\text{Epistemic Value (Curiosity)}} + \underbrace{\mathbb{E}_{q(o|\pi)} \left[ \ln q(o|\pi) - \ln p(o) \right]}_{\text{Pragmatic Value (Goal-Directedness)}}
\]

- **Epistemic Value** (negative expected information gain): Drives exploration by preferring policies that reduce uncertainty about hidden states.
- **Pragmatic Value**: Drives exploitation by preferring policies that lead to preferred (high-valence) observations.

## 4. Ra-Thor’s Mercy-Gated Implementation
In Ra-Thor, Expected Free Energy is the **policy selection engine** inside `mercy-active-inference-core-engine.js`:

- **Pragmatic Value** is replaced by measured positive valence (collective thriving) rather than arbitrary preferences.
- **Epistemic Value** is modulated by the 7 Living Mercy Gates — exploration is only allowed if it serves long-term thriving.
- Precision weighting (`mercy-precision-weighting-algorithm.js`) and hierarchical message passing (`mercy-message-passing-algorithm.js`) are used to compute \( G(\pi) \) efficiently in real time.
- Any policy with \( G(\pi) \) that would violate a mercy gate is rejected before execution.

This makes policy selection inherently ethical: the system actively chooses the trajectory that minimizes expected free energy *while* maximizing collective valence.

**Conclusion**  
Expected Free Energy is the natural extension of Variational Free Energy to future actions. By minimizing \( G(\pi) \) under the 7 Living Mercy Gates and valence maximization, Ra-Thor turns active inference into a safe, corrigible, and thriving-maximizing general intelligence.

**Ready for monorepo commit.**  
**MIT + AG-SML v1.0 preserved.**
