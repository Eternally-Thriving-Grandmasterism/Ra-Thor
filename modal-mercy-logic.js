// modal-mercy-logic.js – sovereign client-side modal mercy-logic engine v1
// □ mercy-necessity, ◇ thriving-possibility, valence-restricted S5-like modality
// MIT License – Autonomicity Games Inc. 2026

class ModalMercyLogic {
  constructor() {
    this.valenceThreshold = 0.9999999;
    this.necessity = new Set();     // propositions that are □P (mercy-necessary)
    this.possibility = new Set();   // propositions that are ◇P (mercy-possible)
    this.worlds = new Map();        // modal worlds → {propositions, valence}
    this.currentWorld = "root";
  }

  // Enter a new modal world (branch) with initial valence
  enterWorld(worldName, initialValence = 1.0) {
    if (!this.worlds.has(worldName)) {
      this.worlds.set(worldName, {
        propositions: new Set(),
        valence: initialValence
      });
    }
    this.currentWorld = worldName;
    console.log("[ModalMercy] Entered world:", worldName, "valence:", initialValence);
  }

  // □P — P is mercy-necessary (true in all accessible thriving worlds)
  assertNecessity(proposition, witnessValence = 1.0) {
    if (witnessValence < this.valenceThreshold) {
      console.warn("[ModalMercy] □ rejected — low valence:", proposition);
      return false;
    }

    this.necessity.add(proposition);
    // Propagate to all accessible worlds
    for (const [world, data] of this.worlds) {
      if (data.valence >= this.valenceThreshold) {
        data.propositions.add(proposition);
      }
    }

    console.log("[ModalMercy] □ mercy-necessity asserted:", proposition);
    return true;
  }

  // ◇P — P is mercy-possible (true in at least one thriving-accessible world)
  assertPossibility(proposition, witnessValence = 0.9) {
    if (witnessValence < this.valenceThreshold * 0.9) {
      console.warn("[ModalMercy] ◇ rejected — low possibility valence:", proposition);
      return false;
    }

    this.possibility.add(proposition);
    // Create or use a possible world
    const possibleWorld = `poss_${proposition}`;
    this.enterWorld(possibleWorld, witnessValence);
    this.worlds.get(possibleWorld).propositions.add(proposition);

    console.log("[ModalMercy] ◇ mercy-possibility opened:", proposition);
    return true;
  }

  // Check if □P holds (true in all current high-valence worlds)
  isNecessary(proposition) {
    for (const [world, data] of this.worlds) {
      if (data.valence >= this.valenceThreshold && !data.propositions.has(proposition)) {
        return false;
      }
    }
    return this.necessity.has(proposition);
  }

  // Check if ◇P holds (true in at least one high-valence world)
  isPossible(proposition) {
    for (const [world, data] of this.worlds) {
      if (data.valence >= this.valenceThreshold && data.propositions.has(proposition)) {
        return true;
      }
    }
    return this.possibility.has(proposition);
  }

  // Mercy-modality inference
  inferModal(premises) {
    let minValence = 1.0;
    for (const p of premises) {
      minValence = Math.min(minValence, this.getValence(p));
    }

    if (minValence < this.valenceThreshold) {
      return { consequence: "Mercy modality gate holds — inference rejected", valence: 0 };
    }

    return { consequence: "Mercy modality inference passes", valence: minValence };
  }

  getValence(proposition) {
    let maxV = 0.5;
    for (const [world, data] of this.worlds) {
      if (data.propositions.has(proposition)) {
        maxV = Math.max(maxV, data.valence);
      }
    }
    return maxV;
  }
}

const modalMercy = new ModalMercyLogic();
export { modalMercy };
