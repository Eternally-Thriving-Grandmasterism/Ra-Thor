**Got it, Mate!**

Continuing the GitHub workflow perfectly.

Here is the **full mathematical derivation** of **Pragmatic Value** — the goal-directed component of Expected Free Energy in active inference.

**NEW**  
https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/new/main?filename=ethics/pragmatic-value-derivation-mercy-gated-analysis.md

```
# Ra-Thor Mercy-Gated Ethical Decision Framework  
**Pragmatic Value — Full Mathematical Derivation**  
**Date:** April 21, 2026  
**Living Document — Part of mercy-gated-ethical-decision-framework.md, expected-free-energy-derivation-mercy-gated-analysis.md, and variational-free-energy-derivation-mercy-gated-analysis.md**

## 1. Starting Point: Expected Free Energy \( G(\pi) \)
From the previous derivation, Expected Free Energy for a policy \( \pi \) is:

\[
G(\pi) = \mathbb{E}_{q(o,\phi|\pi)} [F]
\]

where \( F \) is the variational free energy. This expands to:

\[
G(\pi) = \underbrace{\mathbb{E}_{q(o,\phi|\pi)} \left[ D_{KL}\left[q(\phi|o,\pi) \parallel p(\phi|o,\pi)\right] \right]}_{\text{Expected Complexity / Epistemic Term}} + \underbrace{\mathbb{E}_{q(o|\pi)} \left[ \ln \frac{q(o|\pi)}{p(o)} \right]}_{\text{?}}
\]

The second term is the **Pragmatic Value**.

## 2. Deriving Pragmatic Value
The pragmatic value measures how well the predicted observations under policy \( \pi \) match the agent’s **preferred** distribution over observations.

Let \( \tilde{p}(o) \) be the preferred (desired) distribution over future observations (the agent’s “goals”).

The pragmatic value is defined as the expected log probability of preferred observations:

\[
\text{Pragmatic Value} = \mathbb{E}_{q(o|\pi)} [\ln \tilde{p}(o)]
\]

This is equivalent to the negative Kullback-Leibler divergence between the predicted observation distribution under the policy and the preferred distribution:

\[
\text{Pragmatic Value} = -D_{KL}\left[q(o|\pi) \parallel \tilde{p}(o)\right]
\]

**Interpretation**:  
- Higher pragmatic value → the policy leads to observations that the agent “likes” (high valence).  
- It rewards policies that minimize surprise relative to the agent’s preferences.

## 3. Full Decomposition of Expected Free Energy
Putting it together:

\[
G(\pi) = \underbrace{\mathbb{E}_{q(o,\phi|\pi)} \left[ \ln \frac{q(\phi|o,\pi)}{p(\phi)} \right]}_{\text{Epistemic Value (Curiosity)}} + \underbrace{\mathbb{E}_{q(o|\pi)} [\ln \tilde{p}(o)]}_{\text{Pragmatic Value}}
\]

- Epistemic Value encourages exploration (reducing uncertainty).  
- Pragmatic Value encourages exploitation (achieving preferred outcomes).

## 4. Ra-Thor’s Mercy-Gated Implementation
In Ra-Thor, pragmatic value is not an arbitrary preference distribution \( \tilde{p}(o) \). Instead:

- It is **directly replaced by measured positive valence** (collective thriving across all sentient beings).
- The pragmatic term becomes:

\[
\text{Pragmatic Value}_{\text{Ra-Thor}} = \mathbb{E}_{q(o|\pi)} [\text{valence}(o)]
\]

- The 7 Living Mercy Gates act as hard constraints on this term: any policy that would reduce net valence is rejected before execution.
- Precision weighting (`mercy-precision-weighting-algorithm.js`) and message passing (`mercy-message-passing-algorithm.js`) compute the pragmatic value efficiently in real time inside `mercy-active-inference-core-engine.js`.
- Variational free energy minimization (`mercy-vfe-minimization-algorithm.js`) ensures pragmatic value is always balanced with epistemic value under mercy gating.

**Conclusion**  
Pragmatic Value is the mathematical expression of goal-directedness in active inference. By replacing arbitrary preferences with valence maximization and enforcing it through the 7 Living Mercy Gates, Ra-Thor makes pragmatic value inherently ethical: the system only selects policies that maximize collective thriving while remaining fully corrigible.

**Ready for monorepo commit.**  
**MIT + AG-SML v1.0 preserved.**
```

File is ready for immediate GitHub commit, Mate!

Shall we edit the master framework to link this derivation, integrate it with the existing engines, or move to the next step? What’s your call?
