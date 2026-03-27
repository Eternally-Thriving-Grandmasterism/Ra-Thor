// Ra-Thor Deep Civilization Strategy Engine — City-State Strategies Explored
import { enforceMercyGates, calculateLumenasCI } from '../gaming-lattice-core.js';

const DeepCivStrategyEngine = {
  civ6: {
    // Previous leaders and victoryTypes remain untouched
    leaders: { /* ... previous leaders unchanged ... */ },
    victoryTypes: { /* ... previous victory types unchanged ... */ },

    // NEW: Full City-State Strategies
    cityStates: {
      scientific: {
        name: "Scientific City-States (e.g. Geneva)",
        bonus: "Campus district bonuses + great scientist points",
        mercyGatedStrategy: "Share envoys and research agreements to create universal knowledge abundance — turn every city-state into a global research partner",
        bestVictorySynergy: "Science",
        mercyTip: "Never demand tribute — offer protection and mutual thriving instead",
        lumenasCI: 98
      },
      cultural: {
        name: "Cultural City-States (e.g. Valletta, Mohenjo-Daro)",
        bonus: "Theater square + tourism bonuses",
        mercyGatedStrategy: "Build cultural alliances that spread joy and beauty across the map — share wonders and tourism for universal enlightenment",
        bestVictorySynergy: "Culture",
        mercyTip: "Make every city-state a beacon of shared artistic abundance",
        lumenasCI: 97
      },
      militaristic: {
        name: "Militaristic City-States (e.g. Kumasi)",
        bonus: "Combat experience + unit production",
        mercyGatedStrategy: "Use military bonuses only for liberation and defense — transform city-state armies into protectors of thriving peace",
        bestVictorySynergy: "Domination (mercy path)",
        mercyTip: "Turn every warrior into a guardian of harmony",
        lumenasCI: 94
      },
      religious: {
        name: "Religious City-States (e.g. Yerevan)",
        bonus: "Faith + apostle production",
        mercyGatedStrategy: "Spread faith that promotes joy, non-harm, and universal harmony — create a global religion of abundance",
        bestVictorySynergy: "Religion",
        mercyTip: "Never use religion as a weapon — only as a bridge to shared heavens",
        lumenasCI: 99
      },
      commercial: {
        name: "Commercial City-States (e.g. Amsterdam, Lisbon)",
        bonus: "Trade route + gold bonuses",
        mercyGatedStrategy: "Build infinite RBE-style trade networks that benefit every civilization — create shared economic abundance for the entire world",
        bestVictorySynergy: "Diplomatic / Science",
        mercyTip: "Turn every trade route into a lifeline of mutual prosperity",
        lumenasCI: 96
      },
      industrial: {
        name: "Industrial City-States (e.g. Hattusa)",
        bonus: "Industrial zone + production bonuses",
        mercyGatedStrategy: "Share production capacity to build wonders and infrastructure that uplift all cities — RBE-style industrial abundance",
        bestVictorySynergy: "Science",
        mercyTip: "Turn every factory into a hub of collective thriving",
        lumenasCI: 95
      },
      maritime: {
        name: "Maritime City-States (e.g. Auckland)",
        bonus: "Harbor + naval bonuses",
        mercyGatedStrategy: "Build global trade fleets that connect every continent in peaceful abundance",
        bestVictorySynergy: "Diplomatic",
        mercyTip: "Make every ocean a shared highway of harmony",
        lumenasCI: 93
      }
    }
  },

  generateDeepStrategy(game = "civ6", leader = null, victoryType = null, cityStateType = null, playerLevel = "grandmaster") {
    const base = this[game] || this.civ6;
    
    let strategy = {};
    if (cityStateType && base.cityStates[cityStateType]) {
      strategy = base.cityStates[cityStateType];
    } else if (leader && base.leaders[leader]) {
      strategy = base.leaders[leader];
    } else if (victoryType && base.victoryTypes[victoryType]) {
      strategy = base.victoryTypes[victoryType];
    } else {
      strategy = base.cityStates.scientific; // default to knowledge abundance
    }

    strategy = enforceMercyGates(strategy);
    strategy.lumenasCI = calculateLumenasCI(strategy, playerLevel);

    return {
      game,
      leader,
      victoryType,
      cityStateType,
      strategy,
      offlineShardReady: true,
      message: `Ra-Thor Deep Civilization Lattice™ — mercy-gated ${cityStateType ? 'city-state' : 'general'} strategy`
    };
  }
};

export default DeepCivStrategyEngine;
