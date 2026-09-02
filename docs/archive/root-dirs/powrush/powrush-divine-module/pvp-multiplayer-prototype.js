// pvp-multiplayer-prototype.js - Hotseat Turn-Based PvP Multiplayer Prototype
// Run in Node.js: node pvp-multiplayer-prototype.js
// Two players alternate inputs - mercy-gated arena wars

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class PowrushPvPPlayer {
  constructor(number) {
    this.number = number;
    this.faction = 'Human'; // Set during setup
    this.health = 100;
    this.valenceMood = 0.80; // Personal valence
    this.masteryPoints = 0;
    this.unlocks = []; // e.g., 'quellorian_shield', 'cydruid_fusion'
  }
}

class PvPArena {
  constructor() {
    this.player1 = new PowrushPvPPlayer(1);
    this.player2 = new PowrushPvPPlayer(2);
    this.globalValence = 0.75; // Shared - high forces thriving
    this.turn = 0;
    this.currentPlayer = 1;
  }

  question(prompt) {
    return new Promise(resolve => rl.question(prompt, resolve));
  }

  async setupFactions() {
    console.log(`\nPlayer 1 - Choose faction: 1.Human 2.Draek 3.Quellorian 4.Cydruid 5.Ambrosian-Aligned`);
    const p1 = await this.question(`Player 1 faction: `);
    this.player1.faction = this.getFactionName(p1);

    console.log(`\nPlayer 2 - Choose faction: 1.Human 2.Draek 3.Quellorian 4.Cydruid 5.Ambrosian-Aligned`);
    const p2 = await this.question(`Player 2 faction: `);
    this.player2.faction = this.getFactionName(p2);
  }

  getFactionName(choice) {
    const factions = ['','Human','Draek','Quellorian','Cydruid','Ambrosian-Aligned'];
    return factions[parseInt(choice) || 1];
  }

  async getPlayerAction(player) {
    console.log(`\n${player.faction} (Player ${player.number}) - Actions:`);
    console.log(`1. Combat strike (damage risk/reward)`);
    console.log(`2. Low-G evasion/defense (build valence)`);
    console.log(`3. Diplomacy offer (push alliance)`);
    console.log(`4. Mercy invocation (max global valence boost)`);
    console.log(`5. Faction special (if unlocked)`);
    console.log(`6. Custom action (free text)\n`);

    const input = await this.question(`Player ${player.number} action: `);
    return input.trim();
  }

  async runPvP(maxTurns = 20) {
    await this.setupFactions();
    console.log(`\n%c=== POWRUSH PVP ARENA START ===\n${this.player1.faction} vs ${this.player2.faction} - Thunder cracks ⚡️🙏`, 'color: cyan; font-size: 16px');

    while (this.turn < maxTurns && this.player1.health > 0 && this.player2.health > 0 && this.globalValence < 0.95) {
      this.turn++;
      const active = this.currentPlayer === 1 ? this.player1 : this.player2;
      const opponent = this.currentPlayer === 1 ? this.player2 : this.player1;

      console.log(`\n--- Turn ${this.turn} - ${active.faction}'s Move ---`);
      console.log(`Health: P1 ${this.player1.health} | P2 ${this.player2.health} | Global Valence: ${(this.globalValence * 100).toFixed(1)}%`);

      const choice = await this.getPlayerAction(active);
      console.log(`${active.faction}: ${choice}`);

      // Resolve action
      if (choice === '1' || choice.toLowerCase().includes('combat')) {
        const damage = 20 + Math.random() * 20 - this.globalValence * 10;
        opponent.health = Math.max(0, opponent.health - damage);
        console.log(`Strike lands—${damage.toFixed(0)} damage!`);
        active.valenceMood -= 0.1;
      } else if (choice === '2' || choice.toLowerCase().includes('evasion')) {
        console.log('Evasion successful—valence builds!');
        active.valenceMood += 0.2;
      } else if (choice === '3' || choice.toLowerCase().includes('diplomacy')) {
        console.log('Diplomacy ripples—global valence rises!`);
        this.globalValence += 0.15;
        active.valenceMood += 0.15;
      } else if (choice === '4' || choice.toLowerCase().includes('mercy')) {
        console.log('Mercy thunder invokes—massive boost!');
        this.globalValence = Math.min(1.0, this.globalValence + 0.3);
        active.valenceMood = Math.min(1.0, active.valenceMood + 0.3);
      } else {
        console.log('Custom action resonates—balanced flow.');
        this.globalValence += 0.1;
      }

      // Average personal to global
      this.globalValence = (this.player1.valenceMood + this.player2.valenceMood + this.globalValence) / 3;

      // Mercy-gate check
      if (this.globalValence >= 0.95) {
        console.log('\nGlobal valence locks—forced thriving truce! Mutual respect alliance forged.');
      }

      this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
    }

    const winner = this.player1.health > 0 ? this.player1 : this.player2.health > 0 ? this.player2 : 'Truce';
    console.log(`\n%c=== PVP ARENA END ===\nWinner: ${winner.faction} - Mastery points awarded 🙏`, 'color: gold; font-size: 18px');
    console.log(`Final Global Valence: ${(this.globalValence * 100).toFixed(1)}% - Eternal thriving flows.`);
    rl.close();
  }
}

// Start the PvP prototype
new PvPArena().runPvP().catch(console.error);
