/*!
# Powrush Particle Shaders — PCIe Gen5 DFE Operation

Detailed exploration of Decision Feedback Equalizer (DFE) operation in PCIe Gen5.

## What DFE Does

DFE (Decision Feedback Equalizer) uses decisions from previously received bits to cancel post-cursor inter-symbol interference (ISI) and reflections. It is one of the most powerful equalization techniques at high speeds like 32 GT/s.

Unlike CTLE, DFE does not amplify noise because it uses decided bits in a feedback loop.

## How DFE Operates

1. **Sampling**: The received signal is sampled.
2. **Decision**: A slicer makes a hard decision on the current bit (0 or 1).
3. **Feedback**: The decided bit is fed back through weighted taps to subtract the estimated ISI it would cause on future bits.
4. **Adaptation**: Tap coefficients are continuously or periodically adapted during operation or link training to minimize error.

The feedback effectively "subtracts" the expected interference from previous bits, cleaning up the current sample.

## Multi-Tap DFE

PCIe Gen5 receivers typically use multi-tap DFE (often 1 to 5+ taps):
- Each tap corresponds to a previous bit position (1st post-cursor, 2nd post-cursor, etc.).
- More taps allow cancellation of longer-tail ISI and reflections.

## Interaction with Other Equalizers

DFE works as part of a complete equalization strategy:
- **TX FFE**: Pre-compensates the signal at the transmitter.
- **RX CTLE**: Provides initial high-frequency boost and reduces the burden on DFE.
- **RX DFE**: Removes residual post-cursor ISI that CTLE cannot fully handle.

Good front-end CTLE is important because it reduces the error rate in DFE decisions, minimizing error propagation risk.

## Link Training and Adaptation

During PCIe Gen5 link training, DFE tap coefficients are negotiated and optimized along with TX FFE and RX CTLE settings. Many implementations also support continuous adaptation during normal operation.

## Strengths

- Excellent at removing long-tail ISI and reflections.
- Does not amplify noise or crosstalk.
- Critical for achieving reliable operation at 32 GT/s.

## Challenges

- **Error Propagation**: If a decision error occurs, it can affect subsequent decisions (usually mitigated by good CTLE front-end).
- Complexity and power consumption.
- Requires good adaptation algorithms.

## Relevance to Powrush

For application-level development, DFE operation is hidden inside the SerDes hardware. However, understanding it helps explain why:
- Receiver quality varies significantly between different GPUs and motherboards.
- Some platforms achieve more robust and higher-performance Gen5 operation than others.
- Platform selection matters when trying to maximize PCIe bandwidth for large particle data transfers.

DFE, together with CTLE and TX FFE, is a key reason why high-quality hardware is required to reliably achieve the full benefits of PCIe Gen5.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on DFE operation.
    pub const DFE_OPERATION_NOTES: &str = r#"
        // DFE cancels post-cursor ISI using previous decisions.
        // Works with CTLE and TX FFE in a complete equalization strategy.
        // Multi-tap DFE is standard in Gen5 receivers.
        // Explains receiver quality variation between platforms.
        // Critical for reliable 32 GT/s operation.
    "#;
}
