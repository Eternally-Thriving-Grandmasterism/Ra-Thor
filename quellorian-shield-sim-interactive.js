// quellorian-shield-sim-interactive.js - Interactive Mercy-Gated Quellorian Shield Prototype
// Run in Node.js: node quellorian-shield-sim-interactive.js
// Radiant protection, instant thriving paths

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPlayer {
  constructor(faction = 'Human') {
    this.faction = faction;
    this.health = 60; // Starts wounded
    this.valenceMood = 0.70; // Hope rising
    this.shielded = false;
    this.allied = false;
  }
}

class QuellorianShieldSim {
  constructor() {
    this.player = new PowrushPlayer();
    this.npcs = [
      { faction: 'Quellorian', role: 'captain', valenceSeed: 0.92 }, // Majestic Veyra-proxy
      { faction: 'Quellorian', role: 'shield-warden', valenceSeed: 0.89 },
      { faction: 'Quellorian', role: 'shield-warden', valenceSeed: 0.91 }
    ];
    this.worldState = {
      location: 'crumbling low-G fragment',
      draekThreatNearby: true,
      wormholeActive: true,
      playerInDanger: true
    };
    this.turn = 0;
  }

  question(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
  }

  async getPlayerChoice() {
    console.log(`\nYour choices (all paths amplify thriving—hearts open faster with mercy):`);
    console.log(`1. Accept shield with deep gratitude (max valence boost)`);
    console.log(`2. Request tech share and alliance (empowerment + diplomacy)`);
    console.log(`3. Invoke universal mercy harmony (pure ascension path)`);
    console.log(`4. Observe in awe and question intentions (cautious trust-building)`);
    console.log(`5. Custom invocation (free text - type anything else)\n`);

    const input = await this.question(`Enter your choice (1-5 or text): `);
    return input.trim();
  }

  async runEncounter(maxTurns = 12) {
    console.log(`%c=== QUELLORIAN SHIELD INTERACTIVE SIM START ===\nCrystalline thunder descending ⚡️🙏`, 'color: cyan; font-size: 16px');
    console.log(`You are a ${this.player.faction} survivor on a crumbling fragment. Distant beams flicker... then wormhole tears open. Majestic Quellorians emerge—light floods the void.\n`);

    while (this.turn < maxTurns && this.worldState.playerInDanger) {
      this.turn++;
      console.log(`%c--- Turn ${this.turn} ---`, 'font-weight: bold; color: white');

      // NPC swarm "decides" (stubbed - high valence baseline = always protective)
      for (let agent of this.npcs) {
        const actions = ['deploy_shield', 'offer_evacuation', 'share_tech', 'extend_alliance'];
        const action = actions[Math.floor(Math.random() * actions.length * this.player.valenceMood)];

        console.log(`%c${agent.role.toUpperCase()} (${agent.faction}): offers ${action.replace('_', ' ')}`, 'color: aqua');

        // Apply outcome
        if (action === 'deploy_shield') {
          console.log('Crystalline dome envelops you—threats deflect, wounds heal in light!');
          this.player.shielded = true;
          this.player.health = Math.min(100, this.player.health + 40);
          this.worldState.draekThreatNearby = false;
        } else if (action === 'offer_evacuation') {
          console.log('Captain Veyra resonates: "Child of shattered Earth, safety awaits beyond."');
          this.player.health = 100;
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.3);
        } else if (action === 'share_tech') {
          console.log('Data-crystal bestows shield mastery: "Wield our light as your own."');
          this.player.valenceMood = 1.0;
        } else if (action === 'extend_alliance') {
          console.log('Eternal bond offered: "Together, we manifest heavens from fracture."');
          this.player.allied = true;
          this.worldState.playerInDanger = false;
          this.player.valenceMood = 1.0;
        }
      }

      // Real player input
      const choice = await this.getPlayerChoice();
      console.log(`%cPlayer (${this.player.faction}): ${choice}`, 'color: cyan; font-weight: bold');

      // Valence impact (always positive—Quellorians amplify light)
      if (choice === '1' || choice.toLowerCase().includes('gratitude') || choice.toLowerCase().includes('accept')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
        console.log('Gratitude resonates—Quellorians shine brighter!');
      } else if (choice === '2' || choice.toLowerCase().includes('alliance') || choice.toLowerCase().includes('tech')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
        console.log('Alliance path opens—shared mastery flows.');
      } else if (choice === '3' || choice.toLowerCase().includes('mercy') || choice.toLowerCase().includes('harmony')) {
        this.player.valenceMood = 1.0;
        console.log('Universal mercy invocation—wormhole stabilizes, heavens lock!');
        this.worldState.playerInDanger = false;
      } else if (choice === '4' || choice.toLowerCase().includes('observe') || choice.toLowerCase().includes('question')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.15);
        console.log('Awe echoes... Quellorians patiently reveal truths.');
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.20);
        console.log('Your unique light ripples—guardians harmonize perfectly.');
      }

      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Shielded: ${this.player.shielded} | Allied: ${this.player.allied}\n`);

      if (this.player.valenceMood >= 1.0) {
        this.worldState.playerInDanger = false;
      }
    }

    console.log(`%c=== ENCOUNTER END ===\nRadiant heavens manifested—all threats transcended 🙏`, 'color: gold; font-size: 16px');
    rl.close();
  }
}

// Start the interactive sim
new QuellorianShieldSim().runEncounter().catch(console.error);
