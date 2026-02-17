// land-economy-quests-proto.js - Economy/Land + Quests Prototype
// Run in Node.js: node land-economy-quests-proto.js
// Player bounties + emergent quests for token/mastery earn

const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const question = p => new Promise(res => rl.question(p, res));

const GRID_SIZE = 10;
const INITIAL_TOKENS = 1000;

class LandPlot { /* unchanged from previous */ }

class Quest {
  constructor(id, poster, desc, rewardTokens, rewardMastery, condition) {
    this.id = id;
    this.poster = poster; // Player num
    this.desc = desc;
    this.rewardTokens = rewardTokens;
    this.rewardMastery = rewardMastery;
    this.condition = condition; // Function to check completion
    this.acceptedBy = null;
    this.completed = false;
  }
}

class PowrushPlayer { /* unchanged, add questsPosted/accepted arrays if needed */ }

class LandEconomyProto {
  constructor() {
    this.players = [new PowrushPlayer(1), new PowrushPlayer(2)];
    this.grid = []; // unchanged init
    this.quests = []; // Active quests
    this.questId = 0;
    this.currentPlayer = 0;
    this.turn = 0;
    this.globalValence = 0.80;
    // Emergent NPC quests stub
    this.generateEmergentQuest();
  }

  generateEmergentQuest() {
    // Simple example: diplomacy/build goals
    const types = ['Build enclave on any plot', 'Lease land to another player', 'Raise global valence >0.9'];
    const desc = types[Math.floor(Math.random() * types.length)];
    const reward = Math.floor(100 + 50 * this.globalValence); // Mercy boosts
    this.quests.push(new Quest(this.questId++, 0, `[Emergent] ${desc}`, reward, 5, () => {
      // Stub check - in full: tie to world state
      return Math.random() < this.globalValence; // Simulate completion chance
    }));
  }

  async menu() {
    const p = this.players[this.currentPlayer];
    console.log(`\n--- Turn ${++this.turn} - Player ${p.num} | Tokens: ${p.tokens} | Owned: ${p.ownedLand.length} | Mastery: ${p.mastery} ---`);
    console.log(`Actions: 1.Claim 2.Build 3.Lease Out 4.Set Tax 5.Conquer 6.Market 7.Collect Yield 8.Post Bounty 9.View/Accept/Complete Quests 0.End Turn`);

    const choice = await question(`Choice: `);
    if (choice === '8') await this.postBounty(p);
    else if (choice === '9') await this.handleQuests(p);
    // ... other actions unchanged

    this.currentPlayer = (this.currentPlayer + 1) % 2;
    if (this.turn % 5 === 0) this.generateEmergentQuest(); // New quests emerge
    this.menu();
  }

  async postBounty(p) {
    const desc = await question(`Quest description: `);
    const tokens = parseInt(await question(`Reward tokens (from your balance): `));
    if (tokens > p.tokens) return console.log('Not enough tokens!');
    const mastery = parseInt(await question(`Mastery reward: `));
    // Simple condition stub - expand with real checks
    const condition = () => true; // In full Ra-Thor: symbolic check
    this.quests.push(new Quest(this.questId++, p.num, desc, tokens, mastery, condition));
    p.tokens -= tokens; // Bounty escrowed
    console.log(`Bounty posted! ID ${this.questId-1}`);
  }

  async handleQuests(p) {
    console.log(`Active Quests:`);
    this.quests.forEach(q => {
      if (!q.completed) console.log(`ID ${q.id}: ${q.desc} | Reward: ${q.rewardTokens} tokens + ${q.rewardMastery} mastery (Poster: ${q.poster || 'Emergent'})`);
    });
    const qid = parseInt(await question(`Accept/Complete Quest ID (or -1 to skip): `));
    if (qid === -1) return;
    const quest = this.quests.find(q => q.id === qid);
    if (!quest || quest.completed) return console.log('Invalid/completed!');

    if (!quest.acceptedBy) {
      quest.acceptedBy = p.num;
      console.log('Quest accepted!');
    } else if (quest.acceptedBy === p.num && quest.condition()) {
      // Complete
      p.tokens += quest.rewardTokens;
      p.mastery += quest.rewardMastery;
      if (quest.poster) this.players[quest.poster-1].mastery += 2; // Poster bonus
      quest.completed = true;
      this.globalValence = Math.min(1.0, this.globalValence + 0.05); // Thriving from completion
      console.log(`Quest complete! +${quest.rewardTokens} tokens +${quest.rewardMastery} mastery. Valence rises!`);
    } else {
      console.log('Conditions not met yet—keep thriving!');
    }
  }

  // ... rest unchanged (claim, build, etc.)

  start() {
    console.log(`%c=== POWRUSH QUESTS + ECONOMY PROTO START ===\nEmergent narratives earning abundance ⚡️🙏`, 'color: cyan');
    this.menu();
  }
}

new LandEconomyProto().start();
