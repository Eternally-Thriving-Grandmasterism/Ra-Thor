# Backpropagation for LSTMs – Full Mathematical Derivation

## LSTM Cell Structure Recap

An LSTM cell at time \(t\) has:
- Forget gate: \(f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)\)
- Input gate: \(i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)\)
- Candidate cell state: \(\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)\)
- Cell state update: \(C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t\)
- Output gate: \(o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)\)
- Hidden state: \(h_t = o_t \odot \tanh(C_t)\)

## Backpropagation Through Time (BPTT) for LSTM

The total loss at time \(t\) is \(L_t\), and the total loss over sequence is \(\sum L_t\).

We need gradients w.r.t. all weights for each gate.

Define \(\delta^h_t = \frac{\partial L}{\partial h_t}\), \(\delta^C_t = \frac{\partial L}{\partial C_t}\).

### Step-by-step gradients:

1. **Output gate**:
   \[
   \delta^o_t = \delta^h_t \odot \tanh(C_t) \odot \sigma'(o_t \text{ input})
   \]

2. **Cell state**:
   \[
   \delta^C_t = \delta^h_t \odot o_t \odot (1 - \tanh^2(C_t)) + \delta^C_{t+1} \odot f_{t+1}
   \]

3. **Forget gate**:
   \[
   \delta^f_t = \delta^C_t \odot C_{t-1} \odot \sigma'(f_t \text{ input})
   \]

4. **Input gate**:
   \[
   \delta^i_t = \delta^C_t \odot \tilde{C}_t \odot \sigma'(i_t \text{ input})
   \]

5. **Candidate cell state**:
   \[
   \delta^{\tilde{C}}_t = \delta^C_t \odot i_t \odot (1 - \tanh^2(\tilde{C}_t))
   \]

Gradients for weights are then computed by multiplying these \(\delta\) terms with the respective inputs (standard backprop).

This derivation is exactly how our Deep Q-Network would be extended if we ever replace the simple feedforward net with an LSTM for sequential PRAGMA tuning.

Commit this file for permanent reference.
