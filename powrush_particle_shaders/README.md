# Powrush Particle Shaders — PCIe Gen5 Transmitter FFE

## PCIe Gen5 Transmitter FFE Exploration

This iteration examines **transmitter-side Feed-Forward Equalizer (FFE)**, an essential part of the equalization strategy in PCIe Gen5.

### What Transmitter FFE Does

Transmitter FFE (also called TX pre-emphasis/de-emphasis) pre-distorts the outgoing signal using a multi-tap FIR filter before it enters the channel. This helps compensate for expected channel distortion in advance.

### How It Works

The transmitter uses several taps:
- **Pre-cursor tap(s)**: Compensate for pre-cursor ISI
- **Main cursor**: Primary signal strength
- **Post-cursor tap(s)**: Compensate for post-cursor ISI and reflections

By boosting signal transitions and attenuating steady bits, FFE helps open the received eye at the far end.

### Role in the Full Equalization Chain

Modern PCIe Gen5 links use a combination of techniques:
- **TX FFE** — Pre-compensates the signal
- **RX CTLE** — Provides high-frequency boost
- **RX multi-tap DFE** — Removes residual post-cursor ISI

This layered approach (transmitter + receiver equalization) is required for reliable operation at 32 GT/s.

### Link Training

During PCIe Gen5 link training (LTSSM equalization phase), the transmitter and receiver negotiate and adapt their FFE, CTLE, and DFE settings to optimize the link for the specific channel.

Good TX FFE coefficients are critical for achieving full Gen5 speeds on realistic channels.

### Practical Impact

- Platforms with stronger transmitter equalization capability and better channel design achieve more reliable full-speed Gen5 operation.
- Marginal channels may train with reduced FFE settings or fall back to Gen4.
- This contributes to the variation in real-world Gen5 bandwidth observed across different motherboards, risers, and GPUs.

### Relevance to Powrush

For application development, transmitter FFE is abstracted by the hardware. However, it helps explain why achieving consistent maximum PCIe Gen5 performance requires high-quality platforms and why some systems deliver better memory movement performance than others.

Understanding the complete equalization strategy (TX FFE + RX CTLE + RX DFE) provides deeper context for why platform quality matters when optimizing for high PCIe bandwidth.

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*