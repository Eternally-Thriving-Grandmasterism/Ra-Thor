// pvp-multiplayer-enhanced.js - Enhanced PvP with Specials, Cooldowns, Costs, Progression
// Run in Node.js: node pvp-multiplayer-enhanced.js

const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const question = p => new Promise(res => rl.question(p, res));

class PowrushPvPPlayer {
  constructor(number) {
    this.number = number;
    this.faction = 'Human';
    this.health = 100;
    this.valenceMood = 0.80;
    this.valenceCharge = 50; // Resource for specials (0-100)
    this.masteryPoints = 0;
    this.unlocks = []; // e.g., 'fracture_leap', 'pinnacle_resolve'
    this.cooldowns = {}; // {ability: turnsLeft}
  }

  hasUnlock(ability) { return this.unlocks.includes(ability); }
  onCooldown(ability) { return (this.cooldowns[ability] || 0) > 0; }
}

class PvPArena {
  constructor() {
    this.p1 = new PowrushPvPPlayer(1);
    this.p2 = new PowrushPvPPlayer(2);
    this.globalValence = 0.75;
    this.turn = 0;
    this.current = 1;
  }

  async setup() {
    const factions = ['1.Human','2.Draek','3.Quellorian','4.Cydruid','5.Ambrosian-Aligned'];
    this.p1.faction = this.getFaction(await question(`Player 1 faction (${factions}): `));
    this.p2.faction = this.getFaction(await question(`Player 2 faction (${factions}): `));
  }

  getFaction(choice) {
    const map = {'1':'Human','2':'Draek','3':'Quellorian','4':'Cydruid','5':'Ambrosian-Aligned'};
    return map[choice] || 'Human';
  }

  getSpecials(player) {
    const all = {
      Human: ['fracture_leap','improvise_tool','universal_rally','pinnacle_resolve'],
      Draek: ['neural_stunner','teleport_beam','biomass_scan','pinnacle_doubt'],
      Quellorian: ['deploy_dome','wormhole_beacon','data_crystal','pinnacle_declaration'],
      Cydruid: ['wrist_laser','rune_vine','fissure_dive','pinnacle_weave'],
      'Ambrosian-Aligned': ['subtle_whisper','valence_resonance','void_observation','pinnacle_lattice']
    };
    return all[player.faction] || [];
  }

  async getAction(active, opponent) {
    console.log(`\n${active.faction} (P${active.number}) | Health:${active.health} Charge:${active.valenceCharge} Mastery:${active.masteryPoints}`);
    console.log(`Actions: 1.Combat 2.Evasion 3.Diplomacy 4.Mercy Invocation 5.Special${active.unlocks.length ? ' (Unlocked)' : ''} 6.Custom`);

    const choice = await question(`Player ${active.number} action: `);
    if (choice === '5') {
      const specials = this.getSpecials(active);
      console.log(`Specials (unlocked only):`);
      specials.forEach((s,i) => { if (active.hasUnlock(s)) console.log(`${i+1}.${s} (CD:${active.cooldowns[s]||0})`); });
      const specChoice = parseInt(await question(`Select special (or 0 to cancel): `)) - 1;
      if (specChoice >= 0 && active.hasUnlock(specials[specChoice]) && !active.onCooldown(specials[specChoice])) {
        return {type: 'special', name: specials[specChoice]};
      }
    }
    return {type: choice};
  }

  resolveSpecial(active, opponent, name) {
    const costs = {basic: 30, pinnacle: 60}; // Valence charge cost
    const cds = {basic: 4, pinnacle: 6};
    const isPinnacle = name.includes('pinnacle');
    if (active.valenceCharge < (isPinnacle ? 60 : 30)) {
      console.log('Not enough valence charge!');
      return;
    }
    active.valenceCharge -= (isPinnacle ? 60 : 30);
    active.cooldowns[name] = (isPinnacle ? 6 : 4);

    // Simplified effects (expand in full Ra-Thor)
    if (name.includes('leap') || name.includes('dive')) console.log('Epic reposition—evasion maxed!');
    else if (name.includes('stunner') || name.includes('laser')) { opponent.health -= 30; console.log('Precision strike—30 dmg!'); }
    else if (name.includes('dome') || name.includes('shield')) { active.health += 20; console.log('Protective dome—heal +20!'); }
    else if (name.includes('mercy') || name.includes('declaration') || name.includes('lattice')) { this.globalValence += 0.3; console.log('Massive valence spike—truce nears!'); }
    else console.log(`${name} activates—thunder flows!`);
  }

  async run() {
    await this.setup();
    console.log(`\n%c=== POWRUSH ENHANCED PVP START ===\n${this.p1.faction} vs ${this.p2.faction} ⚡️🙏`, 'color: cyan');

    while (this.turn < 30 && this.p1.health > 0 && this.p2.health > 0 && this.globalValence < 0.95) {
      this.turn++;
      const active = this.current === 1 ? this.p1 : this.p2;
      const opponent = this.current === 1 ? this.p2 : this.p1;

      console.log(`\n--- Turn ${this.turn} - ${active.faction} --- Global Valence: ${(this.globalValence*100).toFixed(1)}%`);

      const action = await this.getAction(active, opponent);

      if (action.type === 'special') this.resolveSpecial(active, opponent, action.name);
      else if (action.type === '1' || action.type.toLowerCase().includes('combat')) { opponent.health -= 20; active.valenceCharge -= 10; }
      else if (action.type === '2') { active.valenceCharge += 15; console.log('Evasion builds charge!'); }
      else if (action.type === '3' || action.type === '4' || action.type.toLowerCase().includes('mercy')) { this.globalValence += 0.2; active.valenceCharge += 30; }
      else { active.valenceCharge += 10; }

      active.valenceCharge = Math.min(100, Math.max(0, active.valenceCharge));
      this.globalValence = (this.p1.valenceCharge + this.p2.valenceCharge)/200 + this.globalValence/3;

      // Cooldown tick
      Object.keys(active.cooldowns).forEach(k => active.cooldowns[k] = Math.max(0, active.cooldowns[k]-1));

      if (this.globalValence >= 0.95) console.log('Global valence locks—thriving truce!');

      this.current = this.current === 1 ? 2 : 1;
    }

    const winner = this.p1.health > 0 ? this.p1 : this.p2.health > 0 ? this.p2 : null;
    if (winner) winner.masteryPoints += this.globalValence >= 0.95 ? 15 : 10;

    // Progression unlocks (example thresholds)
    [this.p1, this.p2].forEach(p => {
      if (p.masteryPoints >= 20 && !p.unlocks.includes(this.getSpecials(p)[0])) p.unlocks.push(this.getSpecials(p)[0]);
      if (p.masteryPoints >= 50 && !p.unlocks.includes(this.getSpecials(p)[3])) p.unlocks.push(this.getSpecials(p)[3]);
    });

    console.log(`\n%c=== ARENA END ===\nMastery earned—specials unlocking 🙏`, 'color: gold');
    rl.close();
  }
}

new PvPArena().run().catch(console.error);
