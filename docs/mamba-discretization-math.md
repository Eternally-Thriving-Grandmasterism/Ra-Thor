# Mamba Discretization Math – Full Derivation (Rathor.ai 2026)

Mamba discretizes the continuous-time State Space Model into a discrete recurrent form for efficient computation.

## Continuous SSM
\[
\mathbf{h}'(t) = A \mathbf{h}(t) + B \mathbf{x}(t)
\]
\[
\mathbf{y}(t) = C \mathbf{h}(t) + D \mathbf{x}(t)
\]

## Discretization (Zero-Order Hold)
The standard discretization uses the matrix exponential:
\[
\overline{A} = \exp(\Delta A)
\]
\[
\overline{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B
\]
\[
\overline{C} = C
\]
\[
\overline{D} = D
\]

where \(\Delta\) is the discretization step size (often learned or fixed).

## Selective Discretization (Mamba Innovation)
Mamba makes \(\Delta\), \(A\), \(B\), and \(C\) **input-dependent**:
\[
\Delta_t = \text{softplus}(W_\Delta \mathbf{x}_t)
\]
\[
\overline{A}_t = \exp(\Delta_t A)
\]
\[
\overline{B}_t = (\Delta_t A)^{-1} (\exp(\Delta_t A) - I) \cdot \Delta_t B_t
\]

This selective mechanism allows the model to focus on relevant parts of the input sequence, enabling linear-time parallel training and RNN-like inference.

This derivation is the mathematical foundation of the Mamba layer used in our Deep Q-Network.

Commit this file for permanent reference.
