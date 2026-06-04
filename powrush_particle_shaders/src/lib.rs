/*!
# Powrush Particle Shaders — PCIe Gen5 Signal Integrity Challenges

Exploration of the physical-layer difficulties introduced by PCIe Gen5.

## Why Gen5 is Harder

PCIe Gen5 runs at 32 GT/s — double the signaling rate of Gen4. At these frequencies, several physical effects that were manageable at lower speeds become severe.

## Major Signal Integrity Challenges

### 1. Insertion Loss
- High-frequency signals attenuate significantly traveling through PCB traces, vias, connectors, and cables.
- Gen5 channels suffer much higher loss than Gen4, reducing eye opening at the receiver.

### 2. Return Loss and Reflections
- Impedance mismatches cause reflections that distort the received signal.
- Requires tighter control over PCB stackup, via design, and connector quality.

### 3. Crosstalk (NEXT/FEXT)
- Adjacent lanes interfere with each other more strongly at higher frequencies.
- Demands better routing practices and shielding.

### 4. Jitter
- Both random jitter (RJ) and deterministic jitter (DJ) become harder to manage.
- Total jitter budget is significantly tighter at 32 GT/s.

### 5. Limited Channel Reach
- Maximum reliable trace length is shorter than Gen4.
- Many legacy designs require retimers, better PCB materials (low-loss laminates), or shorter paths to achieve reliable Gen5 operation.

### 6. Equalization Complexity
- Gen5 requires more sophisticated transmitter and receiver equalization (CTLE + multi-tap DFE + FFE).
- Increases design complexity, power consumption, and silicon area.

### 7. Power and Thermal Impact
- Higher speeds and more complex equalization increase power draw and heat.

### 8. Manufacturing Tolerances
- Smaller margins mean tighter control over fabrication, materials, and assembly is required.
- Small variations can push a link out of spec.

## Practical Impact on GPU Systems

- Many consumer and even some server motherboards struggle to achieve full Gen5 speeds reliably without high-quality components or retimers.
- Real-world bandwidth is often 50-58 GB/s instead of the full 64 GB/s theoretical.
- High-end GPUs and motherboards marketed as "Gen5 ready" still require careful platform design.
- Riser cables and extenders become particularly problematic at Gen5 speeds.

## Relevance to Powrush

Most Powrush development targets typical PCIe systems. Understanding these signal integrity challenges helps explain:
- Why achieving full theoretical Gen5 bandwidth is non-trivial.
- Why some high-end builds still fall back to Gen4 behavior.
- Why careful hardware selection (motherboard, riser quality) matters for maximum performance.

For most workloads, the difference between good Gen4 and good Gen5 is noticeable but not dramatic. However, pushing the limits of PCIe bandwidth makes signal integrity a first-order concern.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on PCIe Gen5 signal integrity.
    pub const PCIE_GEN5_SI_NOTES: &str = r#"
        // Gen5 at 32 GT/s brings serious signal integrity challenges.
        // Insertion loss, crosstalk, jitter, and limited reach are major issues.
        // Real-world systems often fall short of theoretical 64 GB/s.
        // High-quality platforms and careful design are required.
        // Relevant when pushing maximum PCIe bandwidth.
    "#;
}
