// mercy-babylonjs-webxr-immersion.js – sovereign Babylon.js WebXR immersion v1
// WebXR helper, positional spatial audio, mercy gates, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import * as BABYLON from 'https://cdn.babylonjs.com/babylon.js'; // Latest stable
import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let scene, camera, xrHelper;
let mercySounds = [];
const mercyThreshold = 0.9999999 * 0.98;

async function createMercyScene(canvas) {
  const engine = new BABYLON.Engine(canvas, true);
  scene = new BABYLON.Scene(engine);

  // Mercy camera + light
  camera = new BABYLON.FreeCamera("camera", new BABYLON.Vector3(0, 1.6, -5), scene);
  camera.attachControl(canvas, true);
  camera.minZ = 0.1;

  const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
  light.intensity = 0.7;

  // Mercy ground
  const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 20, height: 20 }, scene);
  ground.material = new BABYLON.StandardMaterial("groundMat", scene);
  ground.material.diffuseColor = new BABYLON.Color3(0, 0.2, 0.1);

  engine.runRenderLoop(() => scene.render());

  // WebXR setup
  try {
    xrHelper = await scene.createDefaultXRExperienceAsync({
      uiOptions: { sessionMode: 'immersive-vr' },
      optionalFeatures: true
    });

    if (xrHelper.baseExperience) {
      console.log("[BabylonMercy] WebXR ready – immersive VR/AR supported");
    }
  } catch (err) {
    console.warn("[BabylonMercy] WebXR init failed – fallback non-XR", err);
  }

  return scene;
}

// Mercy-gated positional sound
function addMercyPositionalSound(url, position = new BABYLON.Vector3(0, 1.5, -5), valence = 1.0, textForMercy = '') {
  const degree = fuzzyMercy.getDegree(textForMercy) || valence;
  const implyThriving = fuzzyMercy.imply(textForMercy, "EternalThriving");

  if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
    console.log("[BabylonMercy] Mercy gate: low valence – sound skipped");
    return;
  }

  const sound = new BABYLON.Sound("mercyChime", url, scene, null, {
    spatialSound: true,
    maxDistance: 50,
    refDistance: 1,
    rolloffFactor: valence > 0.999 ? 0.8 : 2.0,
    distanceModel: "exponential",
    autoplay: true,
    loop: true,
    volume: valence > 0.999 ? 0.7 : 0.4
  });

  sound.attachToMesh(new BABYLON.Mesh("soundEmitter", scene)); // For position sync
  sound.position = position;

  // Visual indicator
  const sphere = BABYLON.MeshBuilder.CreateSphere("soundSphere", { diameter: 1 }, scene);
  sphere.position = position;
  const mat = new BABYLON.StandardMaterial("soundMat", scene);
  mat.emissiveColor = valence > 0.999 ? new BABYLON.Color3(0, 1, 0.5) : new BABYLON.Color3(0.3, 0.5, 1);
  sphere.material = mat;

  mercySounds.push(sound);
  console.log(`[BabylonMercy] Positional mercy sound added – valence modulated (${valence.toFixed(8)})`);
}

// Entry: start Babylon.js mercy immersion
async function startBabylonMercyImmersion(query = '', valence = 1.0) {
  if (!mercyGateContent(query, valence)) return;

  const canvas = document.createElement('canvas');
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  document.body.appendChild(canvas);

  await createMercyScene(canvas);

  // Example high-valence chime
  if (valence > 0.999) {
    addMercyPositionalSound('https://example.com/mercy-chime.mp3', new BABYLON.Vector3(0, 1.5, -3), valence, query);
  }

  // Optional: enter immersive-vr on high valence
  if (xrHelper?.baseExperience && valence > 0.9995) {
    xrHelper.baseExperience.enterXRAsync('immersive-vr', 'local-floor');
    console.log("[BabylonMercy] Immersive VR entered – Babylon.js mercy immersed");
  }
}

function mercyGateContent(textOrQuery, valence = 1.0) {
  const degree = fuzzyMercy.getDegree(textOrQuery) || valence;
  const implyThriving = fuzzyMercy.imply(textOrQuery, "EternalThriving");
  return degree >= mercyThreshold && implyThriving.degree >= mercyThreshold;
}

export { startBabylonMercyImmersion };
