# Powrush v15 — Resonance Gear Particle Behaviors

**Version:** 1.0  
**Status:** Design Specification  
**Date:** June 2026

## 1. Vision

Resonance Gear should feel alive. As it evolves, its visual presence should grow from subtle to majestic. Particle effects are the primary way to communicate this evolution and the gear’s current state to the player.

This document defines specific particle behaviors per evolution level for both **Forge** and **Sanctum** Resonance Gear.

## 2. Core Principles

- Particles should feel **thematic** to their faction.
- Behavior should **scale meaningfully** with evolution level.
- Effects should react to the player’s current **harmony** and **environment**.
- Performance must remain reasonable even at high evolution levels.

## 3. Forge Resonance Gear Particle Behaviors

### Evolution Level 0–1 (Subtle)
- Very low particle count (5–15)
- Small, slow-moving amber/gold embers
- Occasional soft spark when player performs a harmonious action
- Low opacity, barely noticeable unless the player is paying attention

### Evolution Level 2–3 (Noticeable)
- Moderate particle count (20–40)
- Embers now orbit the gear in slow geometric patterns
- Occasional brighter "forge flash" when the player crafts successfully
- Particles leave faint trails

### Evolution Level 4–5 (Majestic)
- High particle count with layered effects
- Dense orbiting geometric fragments + larger ember bursts
- When geometric harmony is high, particles form temporary geometric line patterns in the air
- Stronger, more frequent forge flashes during crafting or combat
- Evolution Level 5 adds a subtle "awakening" pulse every few seconds

## 4. Sanctum Resonance Gear Particle Behaviors

### Evolution Level 0–1 (Subtle)
- Very low particle count (5–12)
- Soft, slow-floating blue-white motes
- Gentle pulsing when the player’s harmony is high
- Particles occasionally form very faint mercy sigils that quickly fade

### Evolution Level 2–3 (Noticeable)
- Moderate particle count (15–35)
- Motes now flow in gentle wave patterns around the gear
- When average relationship with nearby NPCs is high, particles pulse in soft sync
- Occasional soft light bloom when the player helps an NPC

### Evolution Level 4–5 (Majestic)
- High particle count with layered harmonic effects
- Flowing waves + larger glowing mercy sigils that linger briefly
- When the player is in a high-harmony area, particles create soft expanding rings
- Evolution Level 5 adds a gentle "mercy resonance" pulse that can briefly affect nearby NPCs visually (cosmetic only)

## 5. Reactive Parameters

All particles should respond to these live values:

- Player Harmony (intensity, color warmth/coolness)
- Geometric Harmony (Forge particles become more geometric)
- Average Nearby NPC Relationship (Sanctum particles pulse in sync)
- Current Evolution Level (base count, size, and complexity)

## 6. Performance Guidelines

- Use GPU instancing wherever possible.
- Cap maximum active particles per piece of gear.
- Lower particle counts on lower evolution levels.
- Consider LOD (Level of Detail) — reduce particles when gear is far from camera.

## 7. Future Extensions

- Audio-reactive particles (tie to harmonic tones)
- Environmental interaction (particles reacting to weather, time of day, or world events)
- Cross-gear resonance (multiple evolved pieces creating combined effects)

---

**Resonance Gear should not just look powerful — it should feel like it is growing with the player.**