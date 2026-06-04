/*!
# Powrush Particle Shaders — DFE Error Propagation Mitigation

Exploration of how error propagation risk in DFE is mitigated in PCIe Gen5.

## The Error Propagation Problem

In DFE, each decided bit is fed back through weighted taps to cancel ISI on future bits. If a decision is incorrect, the wrong feedback value can cause errors on subsequent bits. In severe cases, a single decision error can trigger a burst of errors (error propagation).

This is one of the main challenges of traditional DFE architectures.

## Primary Mitigation: Strong Front-End Equalization

The most effective and widely used mitigation is to **reduce the raw bit error rate entering the DFE** as much as possible:

- **TX FFE + RX CTLE** work together to significantly open the eye *before* the DFE slicer.
- A cleaner input signal to the DFE dramatically lowers the probability of incorrect decisions.
- Lower decision error rate → much lower chance of error propagation.

In well-designed PCIe Gen5 receivers, the combination of TX FFE and CTLE is strong enough that the DFE mostly sees a relatively clean signal, greatly reducing propagation risk.

## Secondary Mitigation Techniques

### Tap Weight Limiting / Clipping
Some DFE implementations limit the maximum magnitude of tap coefficients. This prevents a single erroneous decision from having an excessively large corrective (or disruptive) effect on future samples.

### Protocol-Level Error Recovery
PCIe includes robust error detection (CRC) and retry mechanisms at the Data Link Layer. Even if occasional error bursts occur due to DFE propagation, they are detected and the affected Transaction Layer Packets (TLPs) are retransmitted. This makes rare propagation events tolerable.

### DFE Architecture Enhancements
Some advanced DFE designs use techniques such as:
- Tentative decision DFE
- Reduced-state DFE
- Parallel or look-ahead architectures

These reduce propagation length or probability, but add complexity. Most commercial PCIe Gen5 implementations rely primarily on strong front-end equalization + protocol retry.

## Relevance to Powrush

For application-level work, DFE error propagation and its mitigation are hidden inside the hardware. However, this topic reinforces why:
- High-quality front-end equalization (good CTLE + TX FFE) is critical for reliable Gen5 operation.
- Platform quality (motherboard, GPU, signal integrity) has a real impact on how well the full equalization chain performs.
- Even if occasional errors occur, PCIe’s built-in retry mechanisms provide robust recovery.

Understanding error propagation mitigation helps explain why achieving consistent, high-performance Gen5 links requires both good silicon equalization design *and* good platform-level signal integrity.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on DFE error propagation mitigation.
    pub const DFE_ERROR_PROP_NOTES: &str = r#"
        // Primary mitigation: strong CTLE + TX FFE front-end
        // Reduces raw decision error rate into DFE
        // Secondary: tap limiting + protocol retry (CRC + recovery)
        // Explains why front-end equalization quality is critical
        // Platform signal integrity directly affects propagation risk
    "#;
}
