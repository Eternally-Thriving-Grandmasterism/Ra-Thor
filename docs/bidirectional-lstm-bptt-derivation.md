# Bidirectional LSTM Backpropagation Through Time – Full Derivation

## Architecture Recap

Forward LSTM hidden state: \(\overrightarrow{h}_t\)

Backward LSTM hidden state: \(\overleftarrow{h}_t\)

Combined: \(h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]\)

Output: \(\hat{y}_t = g(W_y h_t + b_y)\)

Total loss: \(L = \sum_t L_t(\hat{y}_t, y_t)\)

## Backpropagation

### Output Layer Gradient
\[
\delta^y_t = \frac{\partial L_t}{\partial \hat{y}_t}
\]

\[
\delta^h_t = W_y^T \delta^y_t
\]

Split into forward and backward parts:
\[
\delta^{\overrightarrow{h}}_t = \delta^h_t[:d], \quad \delta^{\overleftarrow{h}}_t = \delta^h_t[d:]
\]

### Forward LSTM BPTT (backward in time)
Standard LSTM BPTT equations applied to \(\delta^{\overrightarrow{h}}_t\), propagating from \(t\) to \(t-1\).

### Backward LSTM BPTT (forward in time)
Standard LSTM BPTT equations applied to \(\delta^{\overleftarrow{h}}_t\), propagating from \(t\) to \(t+1\).

### Weight Gradients
For any forward weight \(\overrightarrow{W}\):
\[
\frac{\partial L}{\partial \overrightarrow{W}} = \sum_t \delta^{\overrightarrow{h}}_t \frac{\partial \overrightarrow{h}_t}{\partial \overrightarrow{W}}
\]

Same for backward weights.

This derivation enables sequential PRAGMA tuning where past and future context both matter.

Commit this file for permanent reference.
