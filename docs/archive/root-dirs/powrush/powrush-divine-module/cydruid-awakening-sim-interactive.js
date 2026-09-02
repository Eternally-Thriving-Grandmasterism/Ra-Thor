// cydruid-awakening-sim-interactive.js - Interactive Mercy-Gated Cydruid Prototype
// Run in Node.js: node cydruid-awakening-sim-interactive.js
// Builds on PowrushNPCAgent (assume mercy/valencesim logic stubbed for standalone run)

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPlayer {
  constructor(faction = 'Human') {
    this.faction = faction;
    this.health = 75;
    this.valenceMood = 0.78;
    this.fused = false;
  }
}

class CydruidAwakeningSim {
  constructor() {
    this.player = new PowrushPlayer();
    this.npcs = [
      { faction: 'Cydruid', role: 'fissure-leader', valenceSeed: 0.88 },
      { faction: 'Cydruid', role: 'rune-guardian', valenceSeed: 0.85 },
      { faction: 'Cydruid', role: 'rune-guardian', valenceSeed: 0.87 }
    ];
    this.worldState = {
      location: 'volatile low-G fragment',
      fissureActive: true,
      magmaRising: true,
      playerDetected: true
    };
    this.turn = 0;
  }

  question(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
  }

  async getPlayerChoice() {
    console.log(`\nYour choices:`);
    console.log(`1. Accept rune fusion and offer gratitude (max valence boost)`);
    console.log(`2. Request primal tech share (empowerment path)`);
    console.log(`3. Observe in awe and question the guardians (cautious exploration)`);
    console.log(`4. Invoke universal mercy harmony (diplomacy/thriving amplification)`);
    console.log(`5. Custom invocation (free text - type anything else)\n`);

    const input = await this.question(`%cEnter your choice (1-5 or text): `);
    return input.trim();
  }

  async runEncounter(maxTurns = 15) {
    console.log(`%c=== CYDRUID FISSURE AWAKENING INTERACTIVE SIM START ===\nPrimal thunder ascending ⚡️🙏`, 'color: cyan; font-size: 16px');
    console.log(`You are a ${this.player.faction} survivor on a rumbling continent fragment. The ground splits—magma glows below. Ancient voices echo... bio-tech runes ignite.\n`);

    while (this.turn < maxTurns && this.player.health > 0 && this.worldState.fissureActive) {
      this.turn++;
      console.log(`%c--- Turn ${this.turn} ---`, 'font-weight: bold; color: white');

      // NPC swarm "decides" (stubbed evolving actions - in full Ra-Thor, this is Hyperon/NEAT)
      for (let agent of this.npcs) {
        // Simulated mercy-gated action (higher valence = more benevolent)
        const actions = ['bestow_fusion', 'share_primal_tech', 'guide_to_enclave', 'invite_lattice'];
        const action = actions[Math.floor(Math.random() * actions.length * this.player.valenceMood)];

        console.log(`%c${agent.role.toUpperCase()} (${agent.faction}): offers ${action.replace('_', ' ')}`, 'color: lime');

        // Apply outcome
        if (action === 'bestow_fusion') {
          console.log('Rune-light envelops you—ancient Celtic wisdom merges with cyborg lattice. Hybrid power surges!');
          this.player.fused = true;
          this.player.faction = 'Cydruid (Hybrid)';
          this.player.health = 100;
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.3);
        } else if (action === 'share_primal_tech') {
          console.log('Wrist-lasers ignite, earth-runes glow: "Wield the old ways, forged anew."');
          this.player.health = Math.min(100, this.player.health + 40);
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.2);
        } else if (action === 'guide_to_enclave') {
          console.log('Gaia-vines weave a safe path: "Follow the heart\'s thunder."');
          this.worldState.fissureActive = false;
        } else if (action === 'invite_lattice') {
          console.log('Eternal invitation: "Join the grandmaster weave—roots and circuits as one."');
          this.player.valenceMood = 1.0;
        }
      }

      // Real player input
      const choice = await this.getPlayerChoice();
      console.log(`%cPlayer (${this.player.faction}): ${choice}`, 'color: cyan; font-weight: bold');

      // Valence impact from player choice
      if (choice === '1' || choice.toLowerCase().includes('accept') || choice.toLowerCase().includes('fusion') || choice.toLowerCase().includes('gratitude')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
        console.log('Valence surges—guardians resonate stronger!');
      } else if (choice === '2' || choice.toLowerCase().includes('tech') || choice.toLowerCase().includes('share')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.15);
      } else if (choice === '4' || choice.toLowerCase().includes('mercy') || choice.toLowerCase().includes('harmony')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
        console.log('Universal mercy invocation—fissure stabilizes, thriving locks!');
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.10);
        console.log('Your unique path echoes... guardians adapt.');
      }

      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Faction: ${this.player.faction}\n`);
    }

    console.log(`%c=== ENCOUNTER END ===\n${this.player.fused ? 'Hybrid heavens awakened' : 'Roots planted'}—eternal thriving achieved 🙏`, 'color: gold; font-size: 16px');
    rl.close();
  }
}

// Start the interactive sim
new CydruidAwakeningSim().runEncounter().catch(console.error);
