// land-economy-proto.js - Economy & Land Ownership Prototype
// Run in Node.js: node land-economy-proto.js
// Player-driven, sustainable, canon-sealed

const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const question = p => new Promise(res => rl.question(p, res));

const GRID_SIZE = 10; // 10x10 shattered continents
const INITIAL_TOKENS = 1000;

class LandPlot {
  constructor(id, x, y) {
    this.id = id; // NFT-like unique
    this.owner = null; // Player index or null
    this.structure = 'empty'; // empty, enclave, etc.
    this.yield = 0; // Base + structure
    this.lessee = null;
    this.rent = 0;
    this.taxRate = 0; // 0-20%
  }
}

class PowrushPlayer {
  constructor(num) {
    this.num = num;
    this.tokens = INITIAL_TOKENS;
    this.ownedLand = []; // Plot IDs
    this.leasedLand = []; // Plot IDs
    this.mastery = 0;
  }
}

class LandEconomyProto {
  constructor() {
    this.players = [new PowrushPlayer(1), new PowrushPlayer(2)];
    this.grid = [];
    let id = 0;
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        this.grid.push(new LandPlot(id++, x, y));
      }
    }
    this.currentPlayer = 0;
    this.turn = 0;
    this.globalValence = 0.80; // Influences tax caps, conquer odds
  }

  getPlot(id) { return this.grid[id]; }

  async menu() {
    const p = this.players[this.currentPlayer];
    console.log(`\n--- Turn ${++this.turn} - Player ${p.num} | Tokens: ${p.tokens} | Owned: ${p.ownedLand.length} | Mastery: ${p.mastery} ---`);
    console.log(`Actions: 1.Claim Land 2.Build 3.Lease Out 4.Set Tax 5.Conquer 6.Market Trade 7.Collect Yield 8.End Turn`);

    const choice = await question(`Choice: `);
    if (choice === '1') await this.claimLand(p);
    else if (choice === '2') await this.build(p);
    else if (choice === '3') await this.leaseOut(p);
    else if (choice === '4') await this.setTax(p);
    else if (choice === '5') await this.conquer(p);
    else if (choice === '6') await this.market(p);
    else if (choice === '7') this.collectYield();
    // 8 = end turn

    this.currentPlayer = (this.currentPlayer + 1) % 2;
    this.menu();
  }

  async claimLand(p) {
    const unowned = this.grid.filter(plot => !plot.owner);
    if (!unowned.length) return console.log('No free land!');
    const id = parseInt(await question(`Claim plot ID (0-${this.grid.length-1}, unowned: ${unowned.map(u=>u.id).join(',')}): `));
    const plot = this.getPlot(id);
    if (plot.owner) return console.log('Already owned!');
    plot.owner = p.num;
    p.ownedLand.push(id);
    console.log(`Plot ${id} claimed!`);
  }

  async build(p) {
    if (!p.ownedLand.length) return console.log('No land owned!');
    const id = parseInt(await question(`Build on owned ID (${p.ownedLand.join(',')}): `));
    if (!p.ownedLand.includes(id)) return;
    const plot = this.getPlot(id);
    if (p.tokens < 200) return console.log('Need 200 tokens!');
    p.tokens -= 200;
    plot.structure = 'enclave';
    plot.yield = 10;
    console.log(`Enclave built—+10 yield when leased!`);
  }

  async leaseOut(p) {
    const id = parseInt(await question(`Lease out owned ID (${p.ownedLand.join(',')}): `));
    const plot = this.getPlot(id);
    if (plot.lessee) return console.log('Already leased!');
    const rent = parseInt(await question(`Set rent/turn: `));
    plot.rent = rent;
    const lesseeNum = this.currentPlayer + 1; // Simple: other player
    plot.lessee = lesseeNum;
    this.players[lesseeNum-1].leasedLand.push(id);
    console.log(`Leased to Player ${lesseeNum} for ${rent}/turn.`);
  }

  async setTax(p) {
    const id = parseInt(await question(`Set tax on owned ID (${p.ownedLand.join(',')}): `));
    const plot = this.getPlot(id);
    const maxTax = Math.floor(20 * (1 - this.globalValence)); // High valence lowers predatory tax
    const tax = parseInt(await question(`Tax rate 0-${maxTax}%: `));
    if (tax > maxTax) return console.log('Valence caps tax!');
    plot.taxRate = tax / 100;
    console.log(`Tax set: ${tax}%`);
  }

  async conquer(p) {
    const targetId = parseInt(await question(`Conquer plot ID (owned by other): `));
    const plot = this.getPlot(targetId);
    if (!plot.owner || plot.owner === p.num) return;
    const opponent = this.players[plot.owner-1];
    const successChance = 0.5 + (p.mastery - opponent.mastery)/100 + (this.globalValence - 0.8); // Mercy tilts
    if (Math.random() < successChance) {
      plot.owner = p.num;
      opponent.ownedLand = opponent.ownedLand.filter(l => l !== targetId);
      p.ownedLand.push(targetId);
      p.mastery += 5;
      console.log('Conquer success—land seized!');
    } else {
      console.log('Conquer failed—defender holds!');
    }
  }

  async market(p) {
    const id = parseInt(await question(`List/sell owned ID (${p.ownedLand.join(',')} or -1 to buy): `));
    if (id === -1) {
      // Buy logic stub
      const listing = parseInt(await question(`Buy listed ID (stub): `));
      const price = 500; // Example
      if (p.tokens >= price) {
        p.tokens -= price;
        // Transfer ownership...
        console.log('Purchase complete (stub)!');
      }
    } else {
      const price = parseInt(await question(`List price: `));
      console.log(`Plot ${id} listed for ${price} (market stub—next player can buy).`);
    }
  }

  collectYield() {
    this.grid.forEach(plot => {
      if (plot.lessee && plot.yield > 0) {
        const ownerP = this.players[plot.owner-1];
        const lesseeP = this.players[plot.lessee-1];
        const gross = plot.yield + plot.rent;
        lesseeP.tokens -= gross;
        const tax = gross * plot.taxRate;
        ownerP.tokens += gross - tax;
        lesseeP.tokens += plot.yield * (1 - plot.taxRate); // Lessee keeps after-tax yield
        ownerP.mastery += 1; // Passive thriving reward
        console.log(`Yield collected on ${plot.id}: Owner +${gross - tax}, Lessee +${plot.yield * (1 - plot.taxRate)}`);
      }
    });
    this.globalValence = Math.min(1.0, this.globalValence + 0.05); // Economy thrives with activity
  }

  start() {
    console.log(`%c=== POWRUSH ECONOMY & LAND PROTO START ===\nSustainable abundance ascending ⚡️🙏`, 'color: cyan');
    this.menu();
  }
}

new LandEconomyProto().start();
