# Mamba Backpropagation Derivation – Full Mathematical Breakdown (Rathor.ai 2026)

Mamba is a selective State Space Model (SSM). Backpropagation in Mamba follows the standard SSM equations but with input-dependent parameters.

## Core SSM Equations (Continuous)
\[
\mathbf{h}'(t) = A \mathbf{h}(t) + B \mathbf{x}(t)
\]
\[
\mathbf{y}(t) = C \mathbf{h}(t) + D \mathbf{x}(t)
\]

## Discretized Form
\[
\mathbf{h}_t = \overline{A} \mathbf{h}_{t-1} + \overline{B} \mathbf{x}_t
\]
\[
\mathbf{y}_t = \overline{C} \mathbf{h}_t
\]

## Backpropagation Through Time (BPTT)

Define loss \(L\) at each step. The gradient w.r.t. hidden state:
\[
\delta^h_t = \frac{\partial L}{\partial \mathbf{h}_t} = \overline{C}^T \frac{\partial L}{\partial \mathbf{y}_t} + \overline{A}^T \delta^h_{t+1}
\]

Gradients for parameters (selective, input-dependent):
\[
\frac{\partial L}{\partial \overline{A}} = \sum_t \delta^h_t \mathbf{h}_{t-1}^T
\]
\[
\frac{\partial L}{\partial \overline{B}} = \sum_t \delta^h_t \mathbf{x}_t^T
\]
\[
\frac{\partial L}{\partial \overline{C}} = \sum_t \frac{\partial L}{\partial \mathbf{y}_t} \mathbf{h}_t^T
\]

Because parameters are input-dependent, gradients also flow through the selection mechanism (softplus, etc.).

This derivation is the mathematical foundation for training our Mamba-powered DQN.

Commit this file for permanent reference.
