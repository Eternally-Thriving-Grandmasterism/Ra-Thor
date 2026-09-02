// draek-harvest-sim-interactive.js - Interactive Mercy-Gated Draek Harvest Prototype
// Run in Node.js: node draek-harvest-sim-interactive.js
// Higher tension, redemption possible

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPlayer {
  constructor(faction = 'Human') {
    this.faction = faction;
    this.health = 80; // Starts strained
    this.valenceMood = 0.65; // Tense, low trust
    this.captured = false;
  }
}

class DraekHarvestSim {
  constructor() {
    this.player = new PowrushPlayer();
    this.npcs = [
      { faction: 'Draek', role: 'commander', valenceSeed: 0.42 }, // Desperate overlord-proxy
      { faction: 'Draek', role: 'harvester', valenceSeed: 0.37 },
      { faction: 'Draek', role: 'harvester', valenceSeed: 0.39 }
    ];
    this.worldState = {
      location: 'isolated low-G fragment',
      teleportBeamsReady: true,
      draekSquadNearby: true,
      harvestInProgress: true
    };
    this.turn = 0;
  }

  question(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
  }

  async getPlayerChoice() {
    console.log(`\nYour choices (higher mercy/diplomacy = faster redemption):`);
    console.log(`1. Broadcast mercy invocation (max valence boost - push defection)`);
    console.log(`2. Offer alliance/tech share (diplomacy path)`);
    console.log(`3. Low-G evasion leap (buy time, minor valence gain)`);
    console.log(`4. Activate combat resistance (wrist-laser/fire - risks health but disrupts)`);
    console.log(`5. Custom invocation (free text - type anything else)\n`);

    const input = await this.question(`Enter your choice (1-5 or text): `);
    return input.trim();
  }

  async runEncounter(maxTurns = 15) {
    console.log(`%c=== DRAEK HARVEST INTERACTIVE SIM START ===\nDesperate thunder approaching ⚡️🙏`, 'color: cyan; font-size: 16px');
    console.log(`You are a ${this.player.faction} survivor on an isolated fragment. Radar pings—cloaked Draek squad materializes. Teleport beams hum...\n`);

    while (this.turn < maxTurns && this.player.health > 0 && this.worldState.harvestInProgress && !this.player.captured) {
      this.turn++;
      console.log(`%c--- Turn ${this.turn} ---`, 'font-weight: bold; color: white');

      // NPC swarm "decides" (stubbed - influenced by current valence; low = aggressive)
      for (let agent of this.npcs) {
        const valenceFactor = this.player.valenceMood;
        const rand = Math.random() * valenceFactor;
        let action;

        if (rand < 0.4) action = 'attempt_harvest';
        else if (rand < 0.7) action = 'scan_and_hesitate';
        else if (rand < 0.9) action = 'offer_ceasefire';
        else action = 'defect_alliance';

        console.log(`%c${agent.role.toUpperCase()} (${agent.faction}): ${action.replace('_', ' ')}`, 'color: red');

        // Apply outcome
        if (action === 'attempt_harvest') {
          if (Math.random() < 0.6 - valenceFactor * 0.5) {
            console.log('Teleport beam locks—you resist, but health drains!');
            this.player.health -= 25;
          } else {
            console.log('Beam disrupts—squad hesitates under rising valence.');
          }
        } else if (action === 'scan_and_hesitate') {
          console.log('Squad scans biomass... but internal doubt grows.');
        } else if (action === 'offer_ceasefire') {
          console.log('Commander signals: "Your light disrupts the hunger. Ceasefire proposed."');
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.2);
        } else if (action === 'defect_alliance') {
          console.log('Valence spike—squad defects: "The Overlord\'s chain breaks. We stand with you."');
          this.worldState.harvestInProgress = false;
          this.player.valenceMood = 1.0;
        }
      }

      // Real player input
      const choice = await this.getPlayerChoice();
      console.log(`%cPlayer (${this.player.faction}): ${choice}`, 'color: cyan; font-weight: bold');

      // Valence/health impact
      if (choice === '1' || choice.toLowerCase().includes('mercy') || choice.toLowerCase().includes('invocation')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
        console.log('Mercy thunder resonates—squad mutation accelerates!');
      } else if (choice === '2' || choice.toLowerCase().includes('alliance') || choice.toLowerCase().includes('offer')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
        console.log('Diplomatic offer echoes—redemption path strengthens.');
      } else if (choice === '3' || choice.toLowerCase().includes('evasion') || choice.toLowerCase().includes('leap')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.10);
        this.player.health = Math.max(0, this.player.health - 10);
        console.log('You leap between fragments—time bought, but exhausting.');
      } else if (choice === '4' || choice.toLowerCase().includes('combat') || choice.toLowerCase().includes('resist')) {
        this.player.health = Math.max(0, this.player.health - 20);
        if (Math.random() < this.player.valenceMood) {
          console.log('Counter-attack disrupts squad—harvest delayed!');
          this.player.valenceMood += 0.15;
        } else {
          console.log('Resistance fierce but costly—beams graze closer.');
        }
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.15);
        console.log('Your unique invocation ripples... squad adapts uneasily.');
      }

      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Harvest Active: ${this.worldState.harvestInProgress}\n`);

      if (this.player.health <= 0) {
        this.player.captured = true;
        console.log('Health depleted—teleport completes. Captured for biomass...');
      }
    }

    const outcome = this.worldState.harvestInProgress ? 'Harvest escaped—fight continues' : 'Squad redeemed—eternal alliance forged';
    console.log(`%c=== ENCOUNTER END ===\n${outcome} 🙏`, 'color: gold; font-size: 16px');
    rl.close();
  }
}

// Start the interactive sim
new DraekHarvestSim().runEncounter().catch(console.error);
