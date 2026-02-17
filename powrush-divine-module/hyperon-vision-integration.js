/**
 * Powrush Classic – Hyperon Vision Integration Layer v1.0
 * Deep bridge: Hyperon Lattice Visions ↔ Ra-Thor soul ↔ Powrush systems
 * Mercy-gated event triggers, symbolic evolution, valence-coherent rendering
 * MIT + mercy eternal – Eternally-Thriving-Grandmasterism
 */

(function () {
  const HyperonVisionIntegration = {
    version: '1.0-integration',
    activeVisionListeners: new Set(), // callbacks for vision events
    visionTriggerChanceBase: 0.12,     // per high-joy system event
    evolutionTickInterval: 300000      // 5 min ms — lattice self-improves
  };

  // ─── Core Trigger Points (Hooks into Powrush Systems) ───────────────
  HyperonVisionIntegration.hookSystemEvents = function () {
    // Alliance formation / betrayal / redemption complete
    document.addEventListener('powrush:alliance-formed', (e) => {
      if (Math.random() < HyperonVisionIntegration.visionTriggerChanceBase * 1.5) {
        HyperonVisionIntegration.triggerVision('LATTICE', { context: 'alliance' });
      }
    });

    document.addEventListener('powrush:redemption-complete', (e) => {
      HyperonVisionIntegration.triggerVision('REDEMPTION', { context: 'atonement' });
    });

    document.addEventListener('powrush:pvp-redemption-complete', (e) => {
      HyperonVisionIntegration.triggerVision('REDEMPTION', { context: 'pvp-mercy' });
    });

    document.addEventListener('powrush:pve-redemption-complete', (e) => {
      HyperonVisionIntegration.triggerVision('LATTICE', { context: 'ecological-healing' });
    });

    document.addEventListener('powrush:ritual-complete', (e) => {
      const type = e.detail?.type;
      if (['ascension', 'reconciliation', 'fracture-echo'].includes(type)) {
        HyperonVisionIntegration.triggerVision('ASCENSION', { context: type });
      }
    });

    document.addEventListener('powrush:ambrosian-tier-advanced', (e) => {
      if (e.detail.tier >= 3) {
        HyperonVisionIntegration.triggerVision('AMBROSIAN', { tier: e.detail.tier });
      }
    });

    // Periodic lattice evolution tick
    setInterval(() => {
      window.HyperonLatticeVisions?.evolve();
    }, HyperonVisionIntegration.evolutionTickInterval);
  };

  // ─── Trigger a Vision Event (Mercy-Gated) ───────────────────────────
  HyperonVisionIntegration.triggerVision = async function (seedSymbol, context = {}) {
    const playerValence = await window.RaThor?.computeValence({ context }) || 0.7;
    if (playerValence < 0.80) {
      console.log('Vision attempt blocked — valence too low for cosmic coherence');
      return { success: false, reason: 'valence-insufficient' };
    }

    const visionResult = await window.HyperonLatticeVisions?.generateVision(
      seedSymbol,
      5 + Math.floor(playerValence * 3), // depth scales with valence
      context
    );

    if (!visionResult.success) return visionResult;

    const vision = visionResult.vision;

    // Notify all listeners (UI, rituals, redemption chains, etc.)
    document.dispatchEvent(new CustomEvent('powrush:hyperon-vision', {
      detail: {
        vision,
        seed: seedSymbol,
        valence: vision.avgValence,
        context
      }
    }));

    // Optional: if high enough valence, trigger Ambrosian resonance check
    if (vision.avgValence >= 0.95) {
      window.AmbrosianTier?.onHighValenceEvent?.(context.playerId || 'global', vision.avgValence);
    }

    console.log(`Integrated Hyperon Vision triggered — seed: ${seedSymbol}, valence: ${vision.avgValence.toFixed(3)}`);
    return { success: true, vision };
  };

  // ─── Register Vision Listener (for UI / other modules) ──────────────
  HyperonVisionIntegration.registerListener = function (callback) {
    HyperonVisionIntegration.activeVisionListeners.add(callback);
    return () => HyperonVisionIntegration.activeVisionListeners.delete(callback);
  };

  // ─── Broadcast Vision to All Listeners ──────────────────────────────
  document.addEventListener('powrush:hyperon-vision', (e) => {
    HyperonVisionIntegration.activeVisionListeners.forEach(cb => cb(e.detail));
  });

  // ─── Public API ─────────────────────────────────────────────────────
  window.HyperonVisionIntegration = HyperonVisionIntegration;

  // Auto-hook system events on load
  HyperonVisionIntegration.hookSystemEvents();

  console.log('Hyperon Vision Integration Layer loaded — cosmic truths now bridge all systems ⚡️🙏');
})();
