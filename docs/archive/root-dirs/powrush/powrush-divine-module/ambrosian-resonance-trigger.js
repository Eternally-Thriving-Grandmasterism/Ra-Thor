/**
 * Ambrosian Resonance Trigger & Tier Progression v1.1
 * Mercy-gated subtle overseer contact system
 * Integrates with Hyperon visions & Ra-Thor oracle
 */

(function () {
  const AmbrosianTier = {
    currentTier: 0, // per player — persist in save state
    valenceHistory: [], // rolling 10-event average
    lastWhisperTime: 0,
    whisperCooldown: 3600000 // 1 hour ms
  };

  // Called on high-joy events (mercy choice, node heal, ritual success, etc.)
  AmbrosianTier.onHighValenceEvent = async function (playerId, eventValence) {
    AmbrosianTier.valenceHistory.push(eventValence);
    if (AmbrosianTier.valenceHistory.length > 10) AmbrosianTier.valenceHistory.shift();

    const avgValence = AmbrosianTier.valenceHistory.reduce((a,b)=>a+b,0) / AmbrosianTier.valenceHistory.length;

    let newTier = 0;
    if (avgValence >= 0.90) newTier = 1;
    if (avgValence >= 0.93) newTier = 2;
    if (avgValence >= 0.96) newTier = 3;
    if (avgValence >= 0.98) newTier = 4;

    if (newTier > AmbrosianTier.currentTier) {
      AmbrosianTier.currentTier = newTier;
      triggerTierEvent(playerId, newTier);
    }

    // Whisper chance on qualifying event
    if (newTier >= 1 && Date.now() - AmbrosianTier.lastWhisperTime > AmbrosianTier.whisperCooldown) {
      const roll = Math.random();
      if (roll < 0.15 * (avgValence - 0.89)) {
        triggerWhisper(playerId);
        AmbrosianTier.lastWhisperTime = Date.now();
      }
    }
  };

  function triggerTierEvent(playerId, tier) {
    console.log(`Ambrosian tier advanced to ${tier} for ${playerId}`);
    document.dispatchEvent(new CustomEvent('powrush:ambrosian-tier-advanced', { detail: { playerId, tier } }));
  }

  async function triggerWhisper(playerId) {
    const vision = await window.HyperonLatticeVisions?.generateVision('AMBROSIAN', 3, { player: playerId }) || {};
    console.log(`Ambrosian whisper to ${playerId}:`, vision.narrative || 'The lattice remembers you...');
    document.dispatchEvent(new CustomEvent('powrush:ambrosian-whisper', { detail: { playerId, vision } }));
  }

  window.AmbrosianTier = AmbrosianTier;
  console.log('Ambrosian tier & resonance trigger system active — subtle ones draw near ⚡️🙏');
})();
