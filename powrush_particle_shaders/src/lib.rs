/*!
# Powrush Particle Shaders — CTLE and DFE Equalization Techniques

Examination of Continuous Time Linear Equalizer (CTLE) and Decision Feedback Equalizer (DFE) used in high-speed SerDes such as PCIe Gen5.

## Why Equalization is Needed

At 32 GT/s (PCIe Gen5), high-frequency components of the signal are heavily attenuated by the channel (PCB traces, connectors, vias). Without equalization, the received eye is closed and the bit error rate becomes unacceptable.

Equalization compensates for channel loss and inter-symbol interference (ISI).

## CTLE (Continuous Time Linear Equalizer)

**Role**:
- Analog filter at the receiver front-end.
- Provides high-frequency boost (peaking) to compensate for insertion loss.
- Helps restore high-frequency content that was attenuated by the channel.

**Strengths**:
- Relatively simple and low power.
- Fast response.
- Good at compensating frequency-dependent loss.

**Limitations**:
- Amplifies noise and crosstalk along with the signal.
- Cannot fully cancel reflections or long-tail ISI.
- Limited boost range before noise becomes problematic.

## DFE (Decision Feedback Equalizer)

**Role**:
- Uses decisions from previously received bits to cancel post-cursor inter-symbol interference (ISI) and reflections.
- Typically implemented as a feedback filter with one or more taps.

**Strengths**:
- Very effective at removing residual ISI and reflections after CTLE.
- Does not amplify noise (uses decided bits).
- Essential for closing the eye at Gen5 speeds.

**Limitations**:
- More complex and higher power than CTLE.
- Error propagation risk if a decision error occurs (usually mitigated by good front-end CTLE).
- Requires training/adaptation.

## Typical Receiver Architecture (PCIe Gen5)

Modern high-speed receivers combine multiple techniques:
1. **CTLE** – First stage, provides high-frequency boost.
2. **Multi-tap DFE** – Cleans up remaining post-cursor ISI and reflections.
3. **FFE (Feed-Forward Equalizer)** – Often present at the transmitter side to pre-emphasize the signal.

This combination allows reliable operation at 32 GT/s over realistic channels.

## Relevance to Powrush

For application-level development, CTLE and DFE are abstracted by the hardware. However, understanding them helps explain:
- Why PCIe Gen5 is significantly harder to implement reliably than Gen4.
- Why high-quality motherboards, risers, and GPUs are required to achieve good Gen5 performance.
- Why real-world bandwidth often falls short of theoretical maximum on marginal platforms.
- Why signal integrity becomes a first-order concern when pushing PCIe bandwidth limits.

This knowledge provides deeper context for why careful hardware selection and platform design matter when optimizing memory movement performance.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on CTLE and DFE.
    pub const CTLE_DFE_NOTES: &str = r#"
        // CTLE: high-frequency boost, simple but amplifies noise
        // DFE: cancels post-cursor ISI, more powerful but complex
        // Modern receivers combine CTLE + multi-tap DFE
        // Essential for reliable 32 GT/s operation
        // Explains why Gen5 platforms vary significantly in quality
    "#;
}
