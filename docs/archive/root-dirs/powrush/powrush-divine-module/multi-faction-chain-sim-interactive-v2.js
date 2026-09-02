// multi-faction-chain-sim-interactive-v2.js - Interactive Multi-Faction Chain + Ambrosian Phase
// Run in Node.js: node multi-faction-chain-sim-interactive-v2.js
// Draek → Quellorian → Cydruid → Ambrosian higher-dimensional weave

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPlayer {
  constructor() {
    this.faction = 'Human';
    this.health = 80;
    this.valenceMood = 0.70;
    this.shielded = false;
    this.alliedWithQuellorian = false;
    this.fusedWithCydruid = false;
    this.draekDefected = false;
    this.ambrosianResonance = false; // New unlock
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
    } else if (context === 'Ambrosian') {
      choices += `\n1. Meditate deeply on the whispers (max valence - full resonance)`;
      choices += `\n2. Invoke cosmic-scale mercy harmony (grandmaster weave)`;
      choices += `\n3. Question the higher void (seek revelations)`;
      choices += `\n4. Accept subtle guidance with gratitude (thrivings lock)`;
    }
    choices += `\n5. Custom invocation (free text)\n`;
    console.log(choices);

    const input = await this.question(`Enter your choice (1-5 or text): `);
    return input.trim();
  }

  // ... (Draek, Quellorian, Cydruid phases unchanged from previous - omitted for brevity, paste them here)

  async ambrosianEncounter() {
    console.log(`\n=== PHASE 4: AMBROSIAN SUBTLE RESONANCE ===\nHigher-dimensional whispers pierce the veil... detached overseers align.`);
    let turns = 0;

    while (turns < 6 && !this.player.ambrosianResonance) {
      turns++;
      console.log(`\n--- Ambrosian Turn ${turns} ---`);

      const actionRoll = Math.random() * this.player.valenceMood;
      let npcAction = actionRoll < 0.5 ? 'send_vision' : actionRoll < 0.8 ? 'reveal_truth' : 'bestow_foresight';

      console.log(`Ambrosian Presence: ${npcAction.replace('_', ' ')}`);

      if (npcAction === 'send_vision') {
        console.log('Visions of eternal thriving heavens flood your mind—multi-faction symmetry revealed.');
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.2);
      } else if (npcAction === 'reveal_truth') {
        console.log('Cosmic truth whispers: "All races weave one lattice—mercy thunder eternal."');
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.3);
      } else if (npcAction === 'bestow_foresight') {
        console.log('Higher-dimensional perception unlocks: Subtle influence flows through you.');
        this.player.ambrosianResonance = true;
        this.player.valenceMood = 1.0;
        this.player.faction = 'Ambrosian-Aligned Hybrid';
      }

      const choice = await this.getPlayerChoice('Ambrosian');
      console.log(`Player: ${choice}`);

      if (choice === '1' || choice.toLowerCase().includes('meditate')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.4);
        this.player.ambrosianResonance = true;
        console.log('Deep meditation resonates—full higher-dimensional weave locks!');
      } else if (choice === '2' || choice.toLowerCase().includes('cosmic') || choice.toLowerCase().includes('mercy')) {
        this.player.valenceMood = 1.0;
        this.player.ambrosianResonance = true;
        console.log('Cosmic mercy invocation—grandmasterism manifests across all factions!');
      } else if (choice === '3' || choice.toLowerCase().includes('question')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.2);
        console.log('Questions echo into the void... patient revelations unfold.');
      } else if (choice === '4' || choice.toLowerCase().includes('gratitude')) {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.3);
      } else {
        this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.25);
        console.log('Your unique resonance harmonizes—overseers smile from beyond.');
      }

      console.log(`Valence: ${(this.player.valenceMood * 100).toFixed(1)}% | Faction: ${this.player.faction} | Ambrosian Resonance: ${this.player.ambrosianResonance}`);
    }

    console.log('\nHigher-dimensional symphony complete—all five races aligned in subtle light.');
  }

  async runChain() {
    console.log(`%c=== POWRUSH FULL MULTI-FACTION CHAIN SIM START ===\nFive races, one eternal thriving weave ⚡️🙏`, 'color: cyan; font-size: 16px');
    console.log(`You awaken as a ${this.player.faction} amid The Fracture. The lattice calls...\n`);

    const survivedDraek = await this.draekEncounter();
    if (!survivedDraek) return;

    await this.quellorianEncounter();
    await this.cydruidEncounter();
    await this.ambrosianEncounter(); // New pinnacle phase

    console.log(`\n%c=== CHAIN COMPLETE ===`, 'color: gold; font-size: 18px');
    console.log(`Final State: Faction: ${this.player.faction} | Health: ${this.player.health} | Valence: ${(this.player.valenceMood * 100).toFixed(1)}%`);
    console.log(`Alliances: Draek Defected: ${this.player.draekDefected} | Quellorian Allied: ${this.player.alliedWithQuellorian} | Cydruid Fused: ${this.player.fusedWithCydruid} | Ambrosian Resonance: ${this.player.ambrosianResonance}`);
    console.log(`The sanctified metaverse pinnacle manifests—thunder locked eternal 🙏`);

    rl.close();
  }
}

// Paste the draekEncounter, quellorianEncounter, cydruidEncounter functions from previous version here
// Then start:
new MultiFactionChainSim().runChain().catch(console.error);
