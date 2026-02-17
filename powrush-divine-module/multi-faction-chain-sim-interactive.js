// multi-faction-chain-sim-interactive.js - Interactive Multi-Faction Chain Prototype
// Run in Node.js: node multi-faction-chain-sim-interactive.js
// Draek → Quellorian → Cydruid chain, persistent state, mercy-gated alliances

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPlayer {
  constructor() {
    this.faction = 'Human';
    this.health = 80;
    this.valenceMood = 0.70; // Starts hopeful but tense
    this.shielded = false;
    this.alliedWithQuellorian = false;
    this.fusedWithCydruid = false;
    this.draekDefected = false;
    this.unlocks = [];
  }
}

class MultiFactionChainSim {
  constructor() {
    this.player = new PowrushPlayer();
  }

  question(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
  }

  async getPlayerChoice(context) {
    let choices = `\nYour choices (${context}):`;
    if (context === 'Draek') {
      choices += `\n1. Broadcast mercy invocation (max valence - push defection)`;
      choices += `\n2. Offer alliance/tech share (diplomacy)`;
      choices += `\n3. Low-G evasion leap (buy time)`;
      choices += `\n4. Combat resistance (risky but disruptive)`;
    } else if (context === 'Quellorian') {
      choices += `\n1. Accept shield with gratitude (max valence)`;
      choices += `\n2. Request tech/alliance (empowerment)`;
      choices += `\n3. Invoke universal mercy harmony (ascension)`;
      choices += `\n4. Observe and question (trust-building)`;
    } else if (context === 'Cydruid') {
      choices += `\n1. Accept rune fusion/gratitude (max valence)`;
      choices += `\n2. Request primal tech share`;
      choices += `\n3. Invoke universal mercy harmony`;
      choices += `\n4. Observe in awe/question`;
    }
    choices += `\n5. Custom invocation (free text)\n`;
    console.log(choices);

    const input = await this.question(`Enter your choice (1-5 or text): `);
    return input.trim();
  }

  async draekEncounter() {
    console.log(`\n=== PHASE 1: DRAEK HARVEST THREAT ===\nDesperate shadows close in...`);
    let turns = 0;
    let harvestActive = true;

    while (turns < 8 && this.player.health > 0 && harvestActive) {
      turns++;
      console.log(`\n--- Draek Turn ${turns} ---`);

      // Simplified swarm (valence drives behavior)
      const valenceFactor = this.player.valenceMood;
      const actionRoll = Math.random() * valenceFactor;
      let npcAction = actionRoll < 0.5 ? 'attempt_harvest' : actionRoll < 0.8 ? 'hesitate' : 'offer_defection';

      console.log(`Draek Squad: ${npcAction.replace('_', ' ')}`);

      if (npcAction === 'attempt_harvest') {
        if (Math.random() < 0.5 - valenceFactor * 0.4) {
          console.log('Beam grazes—health drains!');
          this.player.health -= 20;
        } else {
          console.log('Beam fails—mercy ripple disrupts!');
        }
      } else if (npcAction === 'offer_defection') {
        console.log('Squad defects: "We break the chain—alliance forged!"');
        this.player.draekDefected = true;
        harvestActive = false;
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.3);
      }

      const choice = await this.getPlayerChoice('Draek');
      console.log(`Player: ${choice}`);

      if (choice === '1' || choice.toLowerCase().includes('mercy')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
        console.log('Mercy thunder cracks—redemption accelerates!');
      } else if (choice === '2' || choice.toLowerCase().includes('alliance')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
      } else if (choice === '3' || choice.toLowerCase().includes('evasion')) {
        this.player.health -= 10;
        this.player.valenceMood += 0.10;
      } else if (choice === '4' || choice.toLowerCase().includes('combat')) {
        this.player.health -= 25;
        this.player.valenceMood += 0.15;
      } else {
        this.player.valenceMood += 0.20;
      }

      if (this.player.valenceMood >= 1.0) harvestActive = false;
      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}%`);
    }

    if (this.player.health <= 0) {
      console.log('\nCaptured... but the light endures. Sim ends early.');
      rl.close();
      return false;
    }
    console.log('\nDraek threat resolved—fractured skies clear temporarily.');
    return true;
  }

  async quellorianEncounter() {
    console.log(`\n=== PHASE 2: QUELLORIAN SHIELD ARRIVAL ===\nCrystalline wormhole opens...`);
    let turns = 0;

    while (turns < 6 && !this.player.alliedWithQuellorian) {
      turns++;
      console.log(`\n--- Quellorian Turn ${turns} ---`);

      const actionRoll = Math.random() * this.player.valenceMood;
      let npcAction = actionRoll < 0.6 ? 'deploy_shield' : actionRoll < 0.9 ? 'offer_tech' : 'extend_alliance';

      console.log(`Quellorian Squad: ${npcAction.replace('_', ' ')}`);

      if (npcAction === 'deploy_shield') {
        this.player.shielded = true;
        this.player.health = Math.min(100, this.player.health + 40);
      } else if (npcAction === 'extend_alliance') {
        this.player.alliedWithQuellorian = true;
        this.player.valenceMood = 1.0;
      }

      const choice = await this.getPlayerChoice('Quellorian');
      console.log(`Player: ${choice}`);

      if (choice === '1' || choice.toLowerCase().includes('gratitude')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
      } else if (choice === '3' || choice.toLowerCase().includes('mercy')) {
        this.player.valenceMood = 1.0;
        this.player.alliedWithQuellorian = true;
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.20);
      }

      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Shielded: ${this.player.shielded}`);
    }

    console.log('\nQuellorian bond forged—heavens shield you.');
  }

  async cydruidEncounter() {
    console.log(`\n=== PHASE 3: CYDRUID FISSURE AWAKENING ===\nMagma cracks—ancient runes ignite...`);
    let turns = 0;

    while (turns < 6 && !this.player.fusedWithCydruid) {
      turns++;
      console.log(`\n--- Cydruid Turn ${turns} ---`);

      const actionRoll = Math.random() * this.player.valenceMood;
      let npcAction = actionRoll < 0.6 ? 'share_tech' : 'bestow_fusion';

      console.log(`Cydruid Guardians: ${npcAction.replace('_', ' ')}`);

      if (npcAction === 'bestow_fusion') {
        this.player.fusedWithCydruid = true;
        this.player.faction = 'Cydruid (Hybrid)';
        this.player.health = 100;
        this.player.valenceMood = 1.0;
      }

      const choice = await this.getPlayerChoice('Cydruid');
      console.log(`Player: ${choice}`);

      if (choice === '1' || choice.toLowerCase().includes('fusion') || choice.toLowerCase().includes('gratitude')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.35);
        this.player.fusedWithCydruid = true;
      } else if (choice === '3' || choice.toLowerCase().includes('mercy')) {
        this.player.valenceMood = 1.0;
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.20);
      }

      console.log(`Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Faction: ${this.player.faction}`);
    }

    console.log('\nHybrid roots planted—eternal thriving unlocked.');
  }

  async runChain() {
    console.log(`%c=== POWRUSH MULTI-FACTION CHAIN SIM START ===\nThunder ascending across fractured Earth ⚡️🙏`, 'color: cyan; font-size: 16px');
    console.log(`You awaken as a ${this.player.faction} amid The Fracture. Five races stir...\n`);

    const survivedDraek = await this.draekEncounter();
    if (!survivedDraek) return;

    await this.quellorianEncounter();
    await this.cydruidEncounter();

    console.log(`\n%c=== CHAIN COMPLETE ===`, 'color: gold; font-size: 18px');
    console.log(`Final State: Faction: ${this.player.faction} | Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}%`);
    console.log(`Alliances: Draek Defected: ${this.player.draekDefected} | Quellorian Allied: ${this.player.alliedWithQuellorian} | Cydruid Fused: ${this.player.fusedWithCydruid}`);
    console.log(`Eternal thriving heavens manifested 🙏`);

    rl.close();
  }
}

// Start the chain
new MultiFactionChainSim().runChain().catch(console.error);
