// cydruid-awakening-sim.js - Mercy-Gated Cydruid Awakening Prototype
// Builds on PowrushNPCAgent from previous

class PowrushPlayer {
  constructor(faction = 'Human') {
    this.faction = faction;
    this.health = 75; // Battered from The Fracture
    this.valenceMood = 0.78; // Awe and hope rising
    this.fused = false; // Cydruid hybrid unlock flag
  }
}

class CydruidAwakeningSim {
  constructor() {
    this.player = new PowrushPlayer();
    this.npcs = [
      new PowrushNPCAgent('Cydruid', 'fissure-leader', 0.88), // Ancient wise seed
      new PowrushNPCAgent('Cydruid', 'rune-guardian', 0.85),
      new PowrushNPCAgent('Cydruid', 'rune-guardian', 0.87)
    ];
    this.worldState = {
      location: 'volatile low-G fragment',
      fissureActive: true,
      magmaRising: true,
      playerDetected: true
    };
    this.turn = 0;
  }

  async runEncounter(maxTurns = 10) {
    console.log(`%c=== CYDRUID FISSURE AWAKENING SIM START ===\nPrimal thunder ascending ⚡️🙏`, 'color: cyan; font-size: 14px');
    console.log(`You are a ${this.player.faction} survivor on a rumbling continent fragment. The ground splits—magma glows below. Ancient voices echo... bio-tech runes ignite.\n`);

    while (this.turn < maxTurns && this.player.health > 0 && this.worldState.fissureActive) {
      this.turn++;
      console.log(`%c--- Turn ${this.turn} ---`, 'font-weight: bold');

      // NPC swarm decides
      for (let agent of this.npcs) {
        let action = await agent.decideAction(this.worldState, [this.player]);
        console.log(`%c${agent.role.toUpperCase()} (${agent.faction}): ${JSON.stringify(action)}`, 'color: lime');

        // Apply action (simplified outcomes)
        if (action.type === 'bestow_fusion') {
          console.log('Rune-light envelops you—ancient Celtic wisdom merges with cyborg lattice. You feel the hybrid power surge!');
          this.player.fused = true;
          this.player.faction = 'Cydruid (Hybrid)';
          this.player.health = 100;
          this.player.valenceMood = 1.0;
        } else if (action.type === 'share_primal_tech') {
          console.log('Wrist-lasers ignite in your palms, earth-runes glow on skin: "Wield the old ways, forged anew."');
          this.player.health = Math.min(100, this.player.health + 40);
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
        } else if (action.type === 'guide_to_enclave' || action.type === 'earth_shield') {
          console.log('Gaia-vines and force-fields weave a path: "Follow the heart\'s thunder to sanctuary."');
          this.worldState.fissureActive = false;
          this.worldState.playerDetected = false;
        } else if (action.type === 'invite_lattice') {
          console.log('Eternal invitation: "Join the grandmaster weave—thrive as one with the roots and circuits."');
          this.worldState.fissureActive = false;
          this.player.valenceMood = 1.0;
        }

        // Player choice injection (simulated—replace with real input)
        let playerChoice = this.simulatePlayerChoice();
        console.log(`%cPlayer (${this.player.faction}): ${playerChoice}`, 'color: cyan');
        if (playerChoice.includes('accept') || playerChoice.includes('fusion') || playerChoice.includes('gratitude')) {
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.18);
          agent.memory.push({ valenceBoost: true });
        } else if (playerChoice.includes('observe') || playerChoice.includes('question')) {
          console.log('You hold back in awe... guardians patiently resonate, sharing visions of eternal thriving.');
          this.player.valenceMood += 0.10;
        }
      }

      // Global valence re-check
      let globalValence = await mercy.valenceCheckDelta(this.worldState);
      console.log(`Global Valence: ${(globalValence * 100).toFixed(1)}%`);
      console.log('Bio-primal joy-locked harmony.\n');
    }

    console.log(`%c=== ENCOUNTER END ===\nHybrid heavens awakened—eternal thriving roots planted 🙏`, 'color: gold; font-size: 14px');
  }

  simulatePlayerChoice() {
    const choices = [
      'Accept rune fusion and gratitude',
      'Request primal tech share',
      'Observe in awe and question the guardians',
      'Invoke universal mercy harmony'
    ];
    return choices[Math.floor(Math.random() * choices.length)];
  }
}

// Run the sim
new CydruidAwakeningSim().runEncounter().catch(console.error);
