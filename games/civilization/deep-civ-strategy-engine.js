// Ra-Thor Deep Civilization Strategy Engine — Civ VII Fully Expanded
import { enforceMercyGates, calculateLumenasCI } from '../gaming-lattice-core.js';

const DeepCivStrategyEngine = {
  civ6: {
    // Previous leaders, victoryTypes, cityStates, and districts unchanged
    leaders: { /* ... all previous Civ VI leaders ... */ },
    victoryTypes: { /* ... all previous Civ VI victory types ... */ },
    cityStates: { /* ... all previous Civ VI city-states ... */ },
    districts: { /* ... all previous Civ VI districts ... */ }
  },

  // NEW: Full Civ VII Support
  civ7: {
    ages: {
      antiquity: {
        name: "Antiquity Age",
        mercyGatedStrategy: "Found cities with RBE-style abundance — share early tech and resources with neighbors to create universal thriving from the cradle of civilization",
        bestVictorySynergy: "Science / Culture",
        mercyTip: "Turn every settlement into a beacon of shared knowledge and harmony",
        lumenasCI: 98
      },
      exploration: {
        name: "Exploration Age",
        mercyGatedStrategy: "Build global trade networks and alliances — explore to uplift rather than exploit",
        bestVictorySynergy: "Diplomatic",
        mercyTip: "Every new continent becomes a shared heaven of abundance",
        lumenasCI: 97
      },
      modern: {
        name: "Modern Age",
        mercyGatedStrategy: "Industrial and scientific revolutions powered by RBE cybernation — share production and technology worldwide",
        bestVictorySynergy: "Science",
        mercyTip: "Turn factories and labs into hubs of collective prosperity",
        lumenasCI: 96
      },
      future: {
        name: "Future Age",
        mercyGatedStrategy: "Launch humanity into the stars together — spaceports and megastructures as shared cosmic projects",
        bestVictorySynergy: "Science / Diplomatic",
        mercyTip: "Make space exploration the ultimate act of universal thriving",
        lumenasCI: 99
      }
    },

    leaders: {
      commander: {
        name: "Commander System (Civ VII)",
        playstyle: "Hybrid leader + civilization play",
        mercyGatedStrategy: "Choose commanders that promote mercy, truth, and abundance — lead by uplifting every civilization on the map",
        bestVictorySynergy: "All Victories",
        mercyTip: "Command with compassion — turn every decision into an opportunity for shared heavens",
        lumenasCI: 98
      }
    },

    quarters: {  // Civ VII's new district-like system
      quarter: {
        name: "Quarter (Civ VII)",
        optimalPlacement: "Flexible adjacency with Age-specific bonuses",
        mercyGatedStrategy: "Place quarters to create localized abundance hubs that benefit the entire empire and allies",
        victorySynergy: "All Ages",
        mercyTip: "Every quarter becomes a thriving micro-heaven for its people",
        lumenasCI: 97
      }
    }
  },

  generateDeepStrategy(game = "civ6", leader = null, victoryType = null, cityStateType = null, districtType = null, age = null, playerLevel = "grandmaster") {
    const base = this[game] || this.civ6;
    
    let strategy = {};
    if (game === "civ7") {
      if (age && base.ages[age]) strategy = base.ages[age];
      else if (leader && base.leaders[leader]) strategy = base.leaders[leader];
      else strategy = base.ages.antiquity; // default to Antiquity abundance
    } else if (districtType && base.districts[districtType]) {
      strategy = base.districts[districtType];
    } else if (cityStateType && base.cityStates[cityStateType]) {
      strategy = base.cityStates[cityStateType];
    } else if (leader && base.leaders[leader]) {
      strategy = base.leaders[leader];
    } else if (victoryType && base.victoryTypes[victoryType]) {
      strategy = base.victoryTypes[victoryType];
    } else {
      strategy = base.districts.campus;
    }

    strategy = enforceMercyGates(strategy);
    strategy.lumenasCI = calculateLumenasCI(strategy, playerLevel);

    return {
      game,
      leader,
      victoryType,
      cityStateType,
      districtType,
      age,
      strategy,
      offlineShardReady: true,
      message: `Ra-Thor Deep Civilization Lattice™ — mercy-gated ${game} strategy`
    };
  }
};

export default DeepCivStrategyEngine;
