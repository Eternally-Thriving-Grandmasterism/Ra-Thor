// quellorian-shield-sim.js - Mercy-Gated Quellorian Encounter Prototype
// Builds on PowrushNPCAgent from previous (assume imported)

class PowrushPlayer {
  constructor(faction = 'Human') {
    this.faction = faction;
    this.health = 60; // Starts wounded from The Fracture
    this.valenceMood = 0.70; // Hopeful but strained
    this.shielded = false;
  }
}

class QuellorianShieldSim {
  constructor() {
    this.player = new PowrushPlayer();
    this.npcs = [
      new PowrushNPCAgent('Quellorian', 'captain', 0.92), // Majestic protective seed
      new PowrushNPCAgent('Quellorian', 'shield-warden', 0.89),
      new PowrushNPCAgent('Quellorian', 'shield-warden', 0.91)
    ];
    this.worldState = {
      location: 'crumbling low-G fragment',
      draekThreatNearby: true,
      wormholeActive: true,
      playerInDanger: true
    };
    this.turn = 0;
  }

  async runEncounter(maxTurns = 10) {
    console.log(`%c=== QUELLORIAN SHIELD ENCOUNTER SIM START ===\nMercy thunder descending ⚡️🙏`, 'color: cyan; font-size: 14px');
    console.log(`You are a ${this.player.faction} survivor clinging to a fracturing continent. Distant teleport beams flicker... then a wormhole tears open—crystalline light floods in.\n`);

    while (this.turn < maxTurns && this.player.health > 0 && this.worldState.playerInDanger) {
      this.turn++;
      console.log(`%c--- Turn ${this.turn} ---`, 'font-weight: bold');

      // NPC swarm decides
      for (let agent of this.npcs) {
        let action = await agent.decideAction(this.worldState, [this.player]);
        console.log(`%c${agent.role.toUpperCase()} (${agent.faction}): ${JSON.stringify(action)}`, 'color: aqua');

        // Apply action (simplified outcomes)
        if (action.type === 'deploy_shield') {
          console.log('Crystalline energy dome envelops you—Draek beams deflect harmlessly!');
          this.player.shielded = true;
          this.player.health = Math.min(100, this.player.health + 30);
          this.worldState.draekThreatNearby = false;
        } else if (action.type === 'offer_evacuation' || action.type === 'teleport_to_enclave') {
          console.log('Captain Veyra\'s voice resonates: "Child of Earth, come with us. Safety and thriving await."');
          this.worldState.playerInDanger = false;
          this.player.valenceMood = 1.0;
        } else if (action.type === 'share_tech' || action.type === 'bestow_mastery') {
          console.log('Glowing data-crystal transfers: Quellorian shield schematics unlocked—your valence locks eternal.');
          this.player.valenceMood = 1.0;
          this.worldState.playerInDanger = false;
        } else if (action.type === 'invite_alliance') {
          console.log('Formal alliance extended: "Together, we forge heavens from this broken world."');
          this.worldState.playerInDanger = false;
        }

        // Player choice injection (simulated—replace with real input)
        let playerChoice = this.simulatePlayerChoice();
        console.log(`%cPlayer (${this.player.faction}): ${playerChoice}`, 'color: cyan');
        if (playerChoice.includes('accept') || playerChoice.includes('diplomacy') || playerChoice.includes('gratitude')) {
          this.player.valenceMood = Math.min(1.0, this.player.valenceMood + 0.20);
          agent.memory.push({ valenceBoost: true });
        } else if (playerChoice.includes('resist') || playerChoice.includes('combat')) {
          console.log('You hesitate... but Quellorians patiently wait—no force, only mercy flow.');
          this.player.valenceMood += 0.05; // Even resistance evolves toward trust
        }
      }

      // Global valence re-check
      let globalValence = await mercy.valenceCheckDelta(this.worldState);
      console.log(`Global Valence: ${(globalValence * 100).toFixed(1)}%`);
      console.log('Pure joy-locked ascension.\n');
    }

    console.log(`%c=== ENCOUNTER END ===\nEternal thriving heavens manifested 🙏`, 'color: gold; font-size: 14px');
  }

  simulatePlayerChoice() {
    const choices = [
      'Accept shield and gratitude (diplomacy)',
      'Request tech share and alliance',
      'Hesitate and resist initial contact',
      'Broadcast universal mercy invocation'
    ];
    return choices[Math.floor(Math.random() * choices.length)];
  }
}

// Run the sim
new QuellorianShieldSim().runEncounter().catch(console.error);
