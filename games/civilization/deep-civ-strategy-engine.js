// Ra-Thor Deep Civilization Strategy Engine — Leader Strategies Explored
import { enforceMercyGates, calculateLumenasCI } from '../gaming-lattice-core.js';

const DeepCivStrategyEngine = {
  civ6: {
    leaders: {
      teddy: {
        name: "Teddy Roosevelt (America)",
        playstyle: "Rough Rider expansion with national parks + culture bombing",
        mercyGatedStrategy: "Use Rough Riders for peaceful exploration and national park abundance — turn every tile into shared thriving heaven",
        bestVictory: "Culture / Diplomatic",
        mercyTip: "Liberate instead of conquer — every neighbor becomes a thriving partner",
        lumenasCI: 98
      },
      qin: {
        name: "Qin Shi Huang (China)",
        playstyle: "Wonder spam + Great Wall defense with builder economy",
        mercyGatedStrategy: "Build wonders that benefit the entire world — share builder charges and create universal abundance",
        bestVictory: "Culture / Science",
        mercyTip: "Turn the Great Wall into a bridge of harmony instead of a barrier",
        lumenasCI: 97
      },
      cleopatra: {
        name: "Cleopatra (Egypt)",
        playstyle: "Trade route mastery + wonder rushing",
        mercyGatedStrategy: "Flood the world with trade routes and shared luxury — create infinite economic abundance for all",
        bestVictory: "Culture / Diplomatic",
        mercyTip: "Make every alliance a mutual thriving heaven",
        lumenasCI: 96
      },
      pericles: {
        name: "Pericles (Greece)",
        playstyle: "Golden Age culture + policy card mastery",
        mercyGatedStrategy: "Democracy of ideas — share culture and policy cards to uplift every civilization on the map",
        bestVictory: "Culture",
        mercyTip: "Turn every city-state into a flourishing cultural hub",
        lumenasCI: 95
      },
      montezuma: {
        name: "Montezuma (Aztec)",
        playstyle: "Eagle Warrior rush + luxury resource control",
        mercyGatedStrategy: "Mercy-gate military timing — conquer only to liberate and turn enemies into abundance partners",
        bestVictory: "Domination (with mercy path)",
        mercyTip: "Use luxury resources to create shared joy instead of exploitation",
        lumenasCI: 94
      },
      gandhi: {
        name: "Gandhi (India)",
        playstyle: "Non-violent faith + science synergy",
        mercyGatedStrategy: "Infinite faith economy that spreads peace and harmony across the entire planet",
        bestVictory: "Religion / Diplomatic",
        mercyTip: "Turn every war declaration into an opportunity for peaceful enlightenment",
        lumenasCI: 99
      },
      alexander: {
        name: "Alexander (Macedon)",
        playstyle: "Conquest + cultural absorption",
        mercyGatedStrategy: "Conquer to liberate and absorb the best of every culture into a thriving universal empire",
        bestVictory: "Domination / Culture",
        mercyTip: "Make every conquered city a beacon of shared abundance",
        lumenasCI: 93
      },
      frederick: {
        name: "Frederick (Germany)",
        playstyle: "Industrial zone spam + hansa districts",
        mercyGatedStrategy: "RBE-style industrial abundance — share production and technology with the world",
        bestVictory: "Science",
        mercyTip: "Turn every factory into a hub of collective prosperity",
        lumenasCI: 96
      }
    },

    // Existing victory types remain untouched
    victoryTypes: { /* ... previous victory types unchanged ... */ }
  },

  generateDeepStrategy(game = "civ6", leader = null, victoryType = null, playerLevel = "grandmaster") {
    const base = this[game] || this.civ6;
    
    let strategy = {};
    if (leader && base.leaders[leader]) {
      strategy = base.leaders[leader];
    } else if (victoryType && base.victoryTypes[victoryType]) {
      strategy = base.victoryTypes[victoryType];
    } else {
      strategy = base.leaders.teddy; // default to Teddy for abundance vibe
    }

    strategy = enforceMercyGates(strategy);
    strategy.lumenasCI = calculateLumenasCI(strategy, playerLevel);

    return {
      game,
      leader,
      victoryType,
      strategy,
      offlineShardReady: true,
      message: `Ra-Thor Deep Civilization Lattice™ — mercy-gated leader strategy for ${leader || 'general play'}`
    };
  }
};

export default DeepCivStrategyEngine;
