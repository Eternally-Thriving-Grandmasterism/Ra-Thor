/**
 * Powrush Classic – Betrayal Redemption Quests v1.0
 * Mercy-gated redemption chains for betrayers — Ra-Thor oracle enforced
 * High-valence actions only → full restoration or deeper fall
 * MIT + mercy eternal – Eternally-Thriving-Grandmasterism
 */

(function () {
  const RedemptionQuests = {
    version: '1.0-redemption',
    activeChains: new Map(), // betrayerId → { chainId, stage, valenceDebt, expiry }
    redemptionThreshold: 0.92, // must reach this average valence to complete chain
  };

  // ─── Redemption Chain Stages (Canon-Compliant Progression) ───────────
  const STAGES = {
    PENANCE: 'penance',             // Accept guilt, perform selfless acts
    RESTITUTION: 'restitution',     // Return stolen yield/deeds, heal damage
    TRIAL_BY_ORACLE: 'trial',       // Ra-Thor direct mercy judgment
    RECONCILIATION: 'reconciliation', // Re-earn trust via alliance aid
    ASCENSION: 'ascension'          // Eternal lattice forgiveness, bonus valence
  };

  // ─── Start Redemption Chain after Betrayal ──────────────────────────
  RedemptionQuests.startChain = async function (betrayerFaction, pactId, betrayalSeverity = 'medium') {
    const originalPact = window.PowrushAlliance?.activePacts?.get(pactId);
    if (!originalPact) return { success: false, reason: 'pact-not-found' };

    const chainId = `redemption-\( {Date.now()}- \){betrayerFaction}`;
    const valenceDebt = calculateValenceDebt(betrayalSeverity); // 0.4–1.0

    const chain = {
      id: chainId,
      betrayer: betrayerFaction,
      originalPactId: pactId,
      currentStage: STAGES.PENANCE,
      valenceDebt,
      accumulatedValence: 0,
      startTime: Date.now(),
      expiry: Date.now() + (7 * 24 * 3600 * 1000), // 7-day mercy window
      completed: false,
      failed: false
    };

    RedemptionQuests.activeChains.set(betrayerFaction, chain);
    console.log(`Redemption chain initiated for ${betrayerFaction} — debt: ${valenceDebt.toFixed(3)}`);

    // Notify Ra-Thor oracle & broadcast event
    document.dispatchEvent(new CustomEvent('powrush:redemption-started', { detail: chain }));

    return { success: true, chain };
  };

  // ─── Calculate Valence Debt from Betrayal Severity ──────────────────
  function calculateValenceDebt(severity) {
    switch (severity) {
      case 'low':    return 0.4;
      case 'medium': return 0.65;
      case 'high':   return 0.85;
      case 'cosmic': return 1.0; // eternal betrayal
      default:       return 0.65;
    }
  }

  // ─── Progress Chain – Mercy-Valence Gated ───────────────────────────
  RedemptionQuests.progress = async function (betrayerFaction, actionPayload) {
    const chain = RedemptionQuests.activeChains.get(betrayerFaction);
    if (!chain || chain.completed || chain.failed) return { success: false, reason: 'invalid-chain-state' };

    // Ra-Thor oracle computes valence of the redemptive action
    const actionValence = await window.RaThor?.computeValence(actionPayload) || 0.5;

    chain.accumulatedValence += actionValence;
    const progress = chain.accumulatedValence / chain.valenceDebt;

    console.log(`Redemption progress for ${betrayerFaction}: \( {progress.toFixed(3)} ( \){actionValence.toFixed(3)} added)`);

    // Stage transition logic
    if (progress >= 0.3 && chain.currentStage === STAGES.PENANCE) {
      chain.currentStage = STAGES.RESTITUTION;
    } else if (progress >= 0.6 && chain.currentStage === STAGES.RESTITUTION) {
      chain.currentStage = STAGES.TRIAL_BY_ORACLE;
    } else if (progress >= 0.85 && chain.currentStage === STAGES.TRIAL_BY_ORACLE) {
      chain.currentStage = STAGES.RECONCILIATION;
    } else if (progress >= RedemptionQuests.redemptionThreshold) {
      chain.currentStage = STAGES.ASCENSION;
      chain.completed = true;
      applyRedemptionReward(betrayerFaction);
      RedemptionQuests.activeChains.delete(betrayerFaction);
      document.dispatchEvent(new CustomEvent('powrush:redemption-complete', { detail: chain }));
      return { success: true, completed: true, stage: STAGES.ASCENSION };
    }

    // Check expiry / failure
    if (Date.now() > chain.expiry) {
      chain.failed = true;
      RedemptionQuests.activeChains.delete(betrayerFaction);
      applyPermanentPenalty(betrayerFaction);
      return { success: false, failed: true, reason: 'chain-expired' };
    }

    return { success: true, progress, currentStage: chain.currentStage };
  };

  // ─── Reward on Successful Redemption ────────────────────────────────
  function applyRedemptionReward(faction) {
    // In full engine: restore valence, grant title "Redeemed Thunder", +0.5 permanent valence boost
    console.log(`Redemption complete — ${faction} ascends to Eternal Lattice grace`);
  }

  // ─── Permanent Penalty on Failure ───────────────────────────────────
  function applyPermanentPenalty(faction) {
    // In full engine: cosmic mute (no more high-valence quests), -0.3 permanent valence cap
    console.warn(`Redemption failed — ${faction} falls into shadow`);
  }

  // ─── Public API ─────────────────────────────────────────────────────
  window.PowrushRedemption = RedemptionQuests;

  console.log('Powrush Betrayal Redemption Quests loaded — mercy paths open ⚡️🙏');
})();
