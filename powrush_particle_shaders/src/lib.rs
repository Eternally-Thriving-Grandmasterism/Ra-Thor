/*!
# Powrush Particle Shaders — PCIe Gen5 Transmitter FFE

Exploration of transmitter-side Feed-Forward Equalizer (FFE) used in PCIe Gen5.

## What is Transmitter FFE?

FFE (Feed-Forward Equalizer) is applied at the **transmitter** side. It pre-distorts the outgoing signal using a finite impulse response (FIR) filter with multiple taps to compensate for expected channel distortion before the signal enters the channel.

It is also commonly called TX pre-emphasis or de-emphasis.

## How FFE Works

The transmitter uses several taps:
- **Pre-cursor tap(s)**: Compensate for pre-cursor ISI
- **Main cursor**: The primary signal strength
- **Post-cursor tap(s)**: Compensate for post-cursor ISI and reflections

By boosting transitions (pre-emphasis) and attenuating steady bits (de-emphasis), FFE helps open the eye at the receiver.

## Role in PCIe Gen5

At 32 GT/s, channel loss is severe. Transmitter FFE works together with receiver equalization:
- **TX FFE** reduces the burden on the receiver by pre-compensating the signal.
- **RX CTLE** provides high-frequency boost.
- **RX multi-tap DFE** cleans up remaining post-cursor ISI.

This combined TX + RX equalization strategy is essential for reliable Gen5 operation.

## Link Training

PCIe Gen5 uses the Link Training and Status State Machine (LTSSM) to negotiate equalization settings. During the equalization phase, the transmitter and receiver adapt their FFE, CTLE, and DFE coefficients to optimize the link.

Good TX FFE settings (determined during training) are critical for achieving full Gen5 speeds on realistic channels.

## Practical Impact

- Platforms with better transmitter equalization capability and channel design achieve more reliable Gen5 performance.
- Marginal channels may train with reduced FFE settings or fall back to Gen4 speeds.
- This is one reason why real-world Gen5 bandwidth varies significantly between different motherboards, risers, and GPU combinations.

## Relevance to Powrush

For application-level work, transmitter FFE is hidden inside the hardware and link training process. However, it helps explain why:
- Achieving consistent full-speed Gen5 operation requires high-quality hardware.
- Some platforms achieve better memory movement performance than others even on the same generation of PCIe.
- Careful platform selection matters when trying to maximize PCIe bandwidth for large particle data transfers.

Understanding the full equalization chain (TX FFE + RX CTLE + RX DFE) provides deeper insight into why PCIe Gen5 is challenging and why platform quality has a real impact on achievable performance.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on transmitter FFE.
    pub const TX_FFE_NOTES: &str = r#"
        // TX FFE pre-compensates the signal before it enters the channel.
        // Works together with RX CTLE + DFE.
        // Critical during Gen5 link training.
        // Explains platform-to-platform variation in Gen5 performance.
        // Relevant when pushing maximum PCIe bandwidth.
    "#;
}
