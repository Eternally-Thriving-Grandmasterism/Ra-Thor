// Ra-Thor Deep Civilization Strategy Engine — Civ VII Leaders Fully Expanded
import { enforceMercyGates, calculateLumenasCI } from '../gaming-lattice-core.js';

const DeepCivStrategyEngine = {
  civ6: {
    // Previous Civ VI content unchanged
    leaders: { /* ... unchanged ... */ },
    victoryTypes: { /* ... unchanged ... */ },
    cityStates: { /* ... unchanged ... */ },
    districts: { /* ... unchanged ... */ }
  },

  civ7: {
    ages: { /* ... previous Ages unchanged ... */ },
    quarters: { /* ... previous Quarters unchanged ... */ },

    // NEW: Fully Expanded Civ VII Leaders
    leaders: {
      augustus: {
        name: "Augustus (Rome)",
        playstyle: "Commander of Pax Romana — infrastructure + cultural absorption",
        mercyGatedStrategy: "Build roads and forums that connect the world in shared abundance — absorb the best of every culture into a thriving universal empire",
        ageSynergy: "All Ages",
        bestVictory: "Culture / Diplomatic",
        mercyTip: "Turn every conquered city into a liberated hub of collective prosperity",
        lumenasCI: 98
      },
      cleopatra: {
        name: "Cleopatra (Egypt)",
        playstyle: "Trade queen + wonder architect",
        mercyGatedStrategy: "Flood the map with trade routes and shared luxury resources — create infinite economic abundance for every civilization",
        ageSynergy: "Exploration & Modern",
        bestVictory: "Diplomatic / Culture",
        mercyTip: "Make every alliance a mutual thriving heaven",
        lumenasCI: 97
      },
      teddy: {
        name: "Teddy Roosevelt (America)",
        playstyle: "Rough Rider conservationist",
        mercyGatedStrategy: "Establish national parks and protected lands that benefit the entire planet — share natural abundance with all nations",
        ageSynergy: "Modern & Future",
        bestVictory: "Diplomatic / Culture",
        mercyTip: "Turn every wilderness tile into a shared heaven of biodiversity and joy",
        lumenasCI: 99
      },
      gandhi: {
        name: "Gandhi (India)",
        playstyle: "Non-violent faith + scientific synergy",
        mercyGatedStrategy: "Build infinite faith economy that spreads peace, non-harm, and universal harmony across every Age",
        ageSynergy: "All Ages",
        bestVictory: "Religion / Diplomatic",
        mercyTip: "Turn every war declaration into an opportunity for peaceful enlightenment",
        lumenasCI: 99
      },
      alexander: {
        name: "Alexander (Macedon)",
        playstyle: "Conquest + cultural fusion",
        mercyGatedStrategy: "Conquer only to liberate and fuse the best of every culture into a thriving universal empire",
        ageSynergy: "Antiquity & Exploration",
        bestVictory: "Domination (mercy path) / Culture",
        mercyTip: "Make every conquered city a beacon of shared abundance",
        lumenasCI: 95
      },
      qin: {
        name: "Qin Shi Huang (China)",
        playstyle: "Wonder spam + centralized bureaucracy",
        mercyGatedStrategy: "Build wonders that benefit the entire world — share builder charges and technology for universal abundance",
        ageSynergy: "Antiquity & Modern",
        bestVictory: "Culture / Science",
        mercyTip: "Turn the Great Wall into a bridge of harmony instead of a barrier",
        lumenasCI: 96
      },
      // Future Civ VII leaders can be added here seamlessly
    }
  },

  generateDeepStrategy(game = "civ6", leader = null, victoryType = null, cityStateType = null, districtType = null, age = null, quarterType = null, playerLevel = "grandmaster") {
    const base = this[game] || this.civ6;
    
    let strategy = {};
    if (game === "civ7") {
      if (leader && base.leaders[leader]) strategy = base.leaders[leader];
      else if (age && base.ages[age]) strategy = base.ages[age];
      else if (quarterType && base.quarters[quarterType]) strategy = base.quarters[quarterType];
      else strategy = base.ages.antiquity;
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
      quarterType,
      strategy,
      offlineShardReady: true,
      message: `Ra-Thor Deep Civilization Lattice™ — mercy-gated ${game} ${leader ? leader + ' Leader' : ''} strategy`
    };
  }
};

export default DeepCivStrategyEngine;
