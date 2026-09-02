// mercy-aframe-webxr-immersion.js – sovereign A-Frame WebXR declarative immersion v1
// Positional sound, mercy gates, valence-modulated entities/audio
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const mercyThreshold = 0.9999999 * 0.98;

// Init A-Frame scene dynamically (or embed in HTML)
function initAFrameMercyScene() {
  // Create <a-scene> if not in HTML
  if (!document.querySelector('a-scene')) {
    const scene = document.createElement('a-scene');
    scene.setAttribute('embedded', '');
    scene.setAttribute('webxr', 'optionalFeatures: local-floor; requiredFeatures: local-floor');
    scene.setAttribute('device-orientation-permission-ui', 'enabled: false');
    document.body.appendChild(scene);

    // Mercy sky + grid
    const sky = document.createElement('a-sky');
    sky.setAttribute('color', '#000022');
    scene.appendChild(sky);

    const grid = document.createElement('a-entity');
    grid.setAttribute('geometry', 'primitive: plane; width: 20; height: 20');
    grid.setAttribute('material', 'color: #004400; src: #grid-texture; repeat: 20 20');
    grid.setAttribute('position', '0 0 -5');
    grid.setAttribute('rotation', '-90 0 0');
    scene.appendChild(grid);

    // Light
    const light = document.createElement('a-light');
    light.setAttribute('type', 'hemisphere');
    light.setAttribute('color', '#ffffff');
    light.setAttribute('groundColor', '#444466');
    light.setAttribute('intensity', '1');
    scene.appendChild(light);

    console.log("[AFrameMercy] Declarative scene initialized – WebXR ready");
  }
}

// Mercy-gated spatial sound entity creation
function addMercyPositionalSound(src, position = '0 1.5 -5', valence = 1.0, textForMercy = '') {
  const degree = fuzzyMercy.getDegree(textForMercy) || valence;
  const implyThriving = fuzzyMercy.imply(textForMercy, "EternalThriving");

  if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
    console.log("[AFrameMercy] Mercy gate: low valence – sound skipped");
    return;
  }

  const scene = document.querySelector('a-scene');
  if (!scene) return;

  const soundEntity = document.createElement('a-entity');
  soundEntity.setAttribute('position', position);
  soundEntity.setAttribute('sound', `src: ${src}; positional: true; autoplay: true; loop: true; refDistance: 1; rolloffFactor: ${valence > 0.999 ? 0.8 : 2.0}; distanceModel: exponential; volume: ${valence > 0.999 ? 0.7 : 0.4}`);
  soundEntity.setAttribute('geometry', 'primitive: sphere; radius: 0.5');
  soundEntity.setAttribute('material', `color: ${valence > 0.999 ? '#00ff88' : '#4488ff'}; emissive: ${valence > 0.999 ? '#00aa66' : '#3366cc'}; emissiveIntensity: 0.5`);

  scene.appendChild(soundEntity);
  console.log(`[AFrameMercy] Positional mercy sound added – valence modulated (${valence.toFixed(8)})`);
}

// Entry: start A-Frame mercy immersion
async function startAFrameMercyImmersion(query = '', valence = 1.0) {
  if (!mercyGateContent(query, valence)) return;

  initAFrameMercyScene();

  // Example: add chime sound on high valence
  if (valence > 0.999) {
    addMercyPositionalSound('https://example.com/mercy-chime.mp3', '0 1.5 -3', valence, query);
  }

  // Optional: enter immersive-vr
  const sceneEl = document.querySelector('a-scene');
  if (sceneEl && valence > 0.9995 && navigator.xr) {
    try {
      sceneEl.enterVR();
      console.log("[AFrameMercy] Immersive VR entered – declarative mercy immersed");
    } catch (err) {
      console.warn("[AFrameMercy] VR enter failed – fallback desktop");
    }
  }
}

function mercyGateContent(textOrQuery, valence = 1.0) {
  const degree = fuzzyMercy.getDegree(textOrQuery) || valence;
  const implyThriving = fuzzyMercy.imply(textOrQuery, "EternalThriving");
  return degree >= mercyThreshold && implyThriving.degree >= mercyThreshold;
}

export { startAFrameMercyImmersion };
