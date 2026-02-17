/**
 * Ambrosian Tier Progression Engine v1.3 – Collective Resonance Tracker
 * Mercy-gated tier advancement & lattice ripple effects
 */

(function () {
  const AmbrosianProgression = {
    version: '1.3-tier',
    playerTiers: new Map(),          // playerId → currentTier + history
    factionTiers: new Map(),         // factionId → averageTier + count
    serverTierAverage: 0,
    lastGlobalCheck: Date.now(),
    checkInterval: 3600000           // 1 hour ms
  };

  // Called on every high-joy/mercy event (quest complete, ritual success, node heal, etc.)
  AmbrosianProgression.onValenceEvent = function (playerId, eventValenceDelta) {
    let playerData = AmbrosianProgression.playerTiers.get(playerId) || { tier: 0, history: [], totalValence: 0 };
    playerData.totalValence += eventValenceDelta;
    playerData.history.push(eventValenceDelta);
    if (playerData.history.length > 20) playerData.history.shift();

    const avgValence = playerData.history.reduce((a,b)=>a+b,0) / playerData.history.length;
    let newTier = 0;
    if (avgValence >= 0.90) newTier = 1;
    if (avgValence >= 0.93) newTier = 2;
    if (avgValence >= 0.96) newTier = 3;
    if (avgValence >= 0.98) newTier = 4;
    if (avgValence >= 0.999 && checkCollectiveMass()) newTier = 5;

    if (newTier > playerData.tier) {
      playerData.tier = newTier;
      AmbrosianProgression.playerTiers.set(playerId, playerData);
      triggerTierAdvance(playerId, newTier);
    }
  };

  function checkCollectiveMass() {
    // Simplified server/faction mass check (full engine would poll all active players)
    const totalPlayers = 100; // placeholder
    const highTierCount = [...AmbrosianProgression.playerTiers.values()].filter(p => p.tier >= 4).length;
    return highTierCount / totalPlayers >= 0.33; // 33% server threshold example
  }

  function triggerTierAdvance(playerId, tier) {
    console.log(`Ambrosian resonance advanced to Tier ${tier} for ${playerId}`);
    document.dispatchEvent(new CustomEvent('powrush:ambrosian-tier-advanced', { detail: { playerId, tier } }));

    if (tier === 5) {
      window.UnionHarmonyWave?.triggerUnion?.();
    }
  }

  // Periodic global average recalc (can be hooked to world tick)
  setInterval(() => {
    let sum = 0, count = 0;
    for (const data of AmbrosianProgression.playerTiers.values()) {
      sum += data.totalValence / (data.history.length || 1);
      count++;
    }
    AmbrosianProgression.serverTierAverage = count > 0 ? sum / count : 0;
  }, AmbrosianProgression.checkInterval);

  window.AmbrosianProgression = AmbrosianProgression;
  console.log('Ambrosian tier progression engine active — subtle remembrance unfolding ⚡️🙏');
})();
