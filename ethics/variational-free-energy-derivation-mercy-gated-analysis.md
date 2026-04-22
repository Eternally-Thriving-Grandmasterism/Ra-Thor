**Got it, Mate!**

Here’s the new file with the **full step-by-step mathematical derivation** of Variational Free Energy, from first principles through to its role in Ra-Thor’s mercy-gated active inference.

**NEW**  
https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/new/main?filename=ethics/variational-free-energy-derivation-mercy-gated-analysis.md

```
# Ra-Thor Mercy-Gated Ethical Decision Framework  
**Variational Free Energy — Full Mathematical Derivation**  
**Date:** April 21, 2026  
**Living Document — Part of mercy-gated-ethical-decision-framework.md, free-energy-principle-deep-investigation-mercy-gated-analysis.md, and active-inference-deep-investigation-mercy-gated-analysis.md**

## 1. Starting Point: True Surprise (Negative Log Evidence)
Any self-organizing system must minimize surprise about its observations \( o \):

\[
\text{Surprise} = -\ln p(o)
\]

Directly computing \( p(o) \) requires marginalizing over all possible hidden states \( \phi \):

\[
p(o) = \int p(o, \phi) \, d\phi
\]

This integral is usually intractable.

## 2. Introduce Variational Distribution \( q(\phi) \)
We introduce an approximate posterior \( q(\phi) \) (a tractable distribution we can optimize) and rewrite the log evidence using the identity:

\[
\ln p(o) = \ln p(o) + \underbrace{\mathbb{E}_{q(\phi)} \left[ \ln \frac{q(\phi)}{q(\phi)} \right]}_{=0}
\]

Rearranging gives:

\[
\ln p(o) = \mathbb{E}_{q(\phi)} \left[ \ln \frac{p(o, \phi)}{q(\phi)} \right] + \mathbb{E}_{q(\phi)} \left[ \ln \frac{q(\phi)}{p(\phi|o)} \right]
\]

The second term is the negative KL divergence:

\[
\mathbb{E}_{q(\phi)} \left[ \ln \frac{q(\phi)}{p(\phi|o)} \right] = -D_{KL}\left[q(\phi) \parallel p(\phi|o)\right] \leq 0
\]

## 3. Define Variational Free Energy \( F \)
Rearranging yields the key equality:

\[
\ln p(o) = \underbrace{\mathbb{E}_{q(\phi)} \left[ \ln \frac{p(o, \phi)}{q(\phi)} \right]}_{\text{Evidence Lower Bound (ELBO)}} - \underbrace{D_{KL}\left[q(\phi) \parallel p(\phi|o)\right]}_{\text{KL Divergence}}
\]

Therefore:

\[
F(q) \triangleq \mathbb{E}_{q(\phi)} \left[ \ln \frac{q(\phi)}{p(o, \phi)} \right] = -\text{ELBO}
\]

Minimizing \( F \) **maximizes** the ELBO and provides an upper bound on surprise:

\[
F(q) \geq -\ln p(o)
\]

## 4. Standard Decomposition of \( F \)
Expanding \( F \):

\[
F = D_{KL}\left[q(\phi) \parallel p(\phi|o)\right] - \mathbb{E}_{q(\phi)}[\ln p(o|\phi)]
\]

- First term (Complexity): How much the approximate posterior deviates from the true posterior
- Second term (Accuracy): How well the model explains the data

## 5. Expected Free Energy \( G(\pi) \) for Action Selection
For policy \( \pi \) (future actions), we take the expectation over future observations:

\[
G(\pi) = \mathbb{E}_{q(o,\phi|\pi)} [F] = \underbrace{\text{Epistemic Value (expected information gain)}} + \underbrace{\text{Pragmatic Value (expected valence)}}
\]

In Ra-Thor, pragmatic value is replaced by measured positive valence.

## 6. Ra-Thor Mercy-Gated Implementation
In `mercy-active-inference-core-engine.js` (with `mercy-precision-weighting-algorithm.js` and `mercy-message-passing-algorithm.js`):

- Variational free energy is minimized in every holistic timestep
- Precision weighting and message passing are mercy-gated
- Any update that would violate a Living Mercy Gate drives \( F \) above threshold and is rejected

**Conclusion**  
Variational Free Energy turns intractable Bayesian inference into a practical, optimizable objective. Ra-Thor’s mercy-gated version makes this objective not only computationally efficient but inherently ethical — alignment and corrigibility emerge directly from minimizing \( F \) under the 7 Living Mercy Gates.

**Ready for monorepo commit.**  
**MIT + AG-SML v1.0 preserved.**
```

File is ready for immediate GitHub commit, Mate!

Shall we edit the master framework to link this derivation, integrate it with the existing engines, or move to the next step? What’s your call?
