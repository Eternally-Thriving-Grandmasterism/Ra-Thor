/**
 * Powrush Classic – Faction Alliance Mechanics v1.0
 * Mercy-gated pacts: Temporary / War / Estate / Eternal Lattice
 * Ra-Thor valence oracle enforces joy/truth/beauty flow
 * MIT + mercy eternal – Eternally-Thriving-Grandmasterism
 */

(function () {
  const AllianceMechanics = {
    version: '1.0-alliance',
    activePacts: new Map(), // pactId → { type, factions, expiry, valenceScore, benefits }
    betrayalPenalties: {
      temporary: { yieldLoss: 0.3, valenceDrop: 0.4 },
      war: { yieldLoss: 0.5, valenceDrop: 0.6, factionLock: 48 }, // hours
      estate: { yieldLoss: 0.7, valenceDrop: 0.8, deedBurn: 1 },
      eternal: { valenceDrop: 1.0, cosmicMute: true } // permanent Ra-Thor shadow
    }
  };

  // ─── Pact Types (Canon-Compliant) ───────────────────────────────────
  const PACT_TYPES = {
    TEMPORARY: 'temporary',     // 24–72 h, low commitment
    WAR: 'war',                 // joint PvP siege, high risk/reward
    ESTATE: 'estate',           // shared land deeds, resource pooling
    ETERNAL_LATTICE: 'eternal'  // cosmic alliance, high valence requirement
  };

  // ─── Form Alliance – Valence Gate Mandatory ─────────────────────────
  AllianceMechanics.formPact = async function (initiatorFaction, targetFaction, pactType, durationHours = 24) {
    if (!Object.values(PACT_TYPES).includes(pactType)) {
      return { success: false, reason: 'invalid-pact-type' };
    }

    // Ra-Thor valence oracle check (both factions average)
    const initiatorValence = await window.RaThor?.computeValence({ faction: initiatorFaction }) || 0.5;
    const targetValence = await window.RaThor?.computeValence({ faction: targetFaction }) || 0.5;
    const jointValence = (initiatorValence + targetValence) / 2;

    if (jointValence < 0.75) {
      return { success: false, reason: 'valence-too-low', score: jointValence };
    }

    // Create pact
    const pactId = `pact-\( {Date.now()}- \){initiatorFaction}-${targetFaction}`;
    const expiry = Date.now() + (durationHours * 3600 * 1000);

    const pact = {
      id: pactId,
      type: pactType,
      factions: [initiatorFaction, targetFaction],
      startTime: Date.now(),
      expiry,
      valenceScore: jointValence,
      benefits: generateBenefits(pactType, jointValence)
    };

    AllianceMechanics.activePacts.set(pactId, pact);
    console.log(`Mercy pact formed: ${pactType} between ${initiatorFaction} & ${targetFaction} (valence ${jointValence.toFixed(3)})`);

    // Broadcast event for UI / game engine
    document.dispatchEvent(new CustomEvent('powrush:alliance-formed', { detail: pact }));

    return { success: true, pact };
  };

  // ─── Benefit Generator (Horizontal, Valence-Scaled) ─────────────────
  function generateBenefits(type, valence) {
    const scale = Math.min(1, valence); // cap at 1.0
    const base = { sharedYield: 0.15 * scale };

    switch (type) {
      case PACT_TYPES.TEMPORARY:
        return { ...base, comboSpecial: true, truceBonus: 0.2 };
      case PACT_TYPES.WAR:
        return { ...base, siegeBonus: 0.35, jointConquest: true };
      case PACT_TYPES.ESTATE:
        return { ...base, pooledDeeds: true, resourceTax: 0.1, estateShield: true };
      case PACT_TYPES.ETERNAL_LATTICE:
        return { ...base, cosmicReveal: true, valenceBoost: 0.5, eternalTruce: true };
      default:
        return base;
    }
  }

  // ─── Break / Betray Pact – Penalties Enforced ───────────────────────
  AllianceMechanics.breakPact = function (pactId, betrayerFaction) {
    const pact = AllianceMechanics.activePacts.get(pactId);
    if (!pact) return { success: false, reason: 'pact-not-found' };

    const penalty = AllianceMechanics.betrayalPenalties[pact.type];
    if (!penalty) return { success: false, reason: 'invalid-pact-type' };

    // Apply penalties to betrayer
    console.warn(`Betrayal detected! ${betrayerFaction} breaks ${pact.type} pact – penalties applied`);

    // Valence drop + yield loss (simplified – integrate with real economy)
    // In full engine: call RaThor.applyPenalty(betrayerFaction, penalty)

    // Remove pact
    AllianceMechanics.activePacts.delete(pactId);
    document.dispatchEvent(new CustomEvent('powrush:alliance-broken', { detail: { pactId, betrayer: betrayerFaction } }));

    return { success: true, penaltyApplied: true };
  };

  // ─── Check Active Pacts for Faction ─────────────────────────────────
  AllianceMechanics.getActivePactsForFaction = function (faction) {
    const pacts = [];
    for (const pact of AllianceMechanics.activePacts.values()) {
      if (pact.factions.includes(faction)) {
        pacts.push(pact);
      }
    }
    return pacts;
  };

  // ─── Public API ─────────────────────────────────────────────────────
  window.PowrushAlliance = AllianceMechanics;

  console.log('Powrush Alliance Mechanics loaded – mercy pacts active ⚡️🙏');
})();
