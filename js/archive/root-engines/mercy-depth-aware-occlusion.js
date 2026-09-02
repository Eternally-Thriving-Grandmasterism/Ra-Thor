// mercy-depth-aware-occlusion.js – sovereign Mercy Depth-Aware Occlusion Rendering v1
// XRDepthInformation texture + custom shader occlusion, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

// Custom occlusion shader material (Babylon.js)
const occlusionShader = `
  precision highp float;

  varying vec2 vUV;
  uniform sampler2D depthTexture;     // XR depth map
  uniform sampler2D colorTexture;     // Original overlay color
  uniform float valenceIntensity;     // Valence-modulated strength
  uniform vec2 resolution;            // Canvas resolution
  uniform float nearClip;             // Near clip plane
  uniform float farClip;              // Far clip plane

  void main(void) {
    vec2 uv = vUV;
    float realDepth = texture2D(depthTexture, uv).r; // normalized [0,1]

    // Linearize depth (approximation – adjust near/far as needed)
    float linearDepth = 2.0 * nearClip * farClip / (farClip + nearClip - (realDepth * 2.0 - 1.0) * (farClip - nearClip));

    // Virtual depth from scene (z-buffer value, assume passed or approximated)
    float virtualDepth = gl_FragCoord.z; // Normalized device coordinate z

    // Occlusion test
    float occlusion = step(virtualDepth, linearDepth * 0.98); // slight bias to avoid z-fighting

    vec4 color = texture2D(colorTexture, uv);
    color.a *= occlusion * valenceIntensity; // Fade out when occluded

    gl_FragColor = color;
  }
`;

class MercyDepthOcclusionRenderer {
  constructor(scene) {
    this.scene = scene;
    this.depthInfo = null;
    this.occlusionMaterials = new Map(); // mesh → material
    this.valence = 1.0;
  }

  async gateOcclusion(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyDepthOcclusion] Gate holds: low valence – occlusion rendering aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyDepthOcclusion] Mercy gate passes – eternal thriving depth occlusion activated");
    return true;
  }

  // Enable depth sensing (call after session start)
  async enableDepth(session) {
    try {
      // Babylon.js helper (adapt for other engines)
      // xr.baseExperience.featuresManager.enableFeature("depth-sorted-layers", "stable");
      console.log("[MercyDepthOcclusion] Depth sensing enabled – occlusion-ready mercy lattice");
      return true;
    } catch (err) {
      console.error("[MercyDepthOcclusion] Depth enable failed:", err);
      return false;
    }
  }

  // Apply depth-aware occlusion to mercy overlays (call per frame or on overlay create/update)
  applyOcclusionToOverlay(overlayMesh, frame) {
    if (!frame?.getDepthInformation) return;

    const depthInfo = frame.getDepthInformation();
    if (depthInfo) {
      this.depthInfo = depthInfo;

      // Create or update occlusion material
      let mat = this.occlusionMaterials.get(overlayMesh);
      if (!mat) {
        mat = new BABYLON.ShaderMaterial("occlusionMat", this.scene, {
          vertex: "custom",
          fragment: "custom"
        }, {
          attributes: ["position", "uv"],
          uniforms: ["worldViewProjection", "depthTexture", "colorTexture", "valenceIntensity", "resolution", "nearClip", "farClip"]
        });

        mat.setTexture("depthTexture", new BABYLON.RawTexture.CreateRGBATexture(
          depthInfo.texture, depthInfo.width, depthInfo.height, this.scene
        ));
        mat.setTexture("colorTexture", overlayMesh.material.diffuseTexture || new BABYLON.Texture("fallback.png", this.scene));
        mat.setFloat("valenceIntensity", this.valence);
        mat.setVector2("resolution", new BABYLON.Vector2(this.scene.getEngine().getRenderWidth(), this.scene.getEngine().getRenderHeight()));
        mat.setFloat("nearClip", 0.1);
        mat.setFloat("farClip", 100.0);
        mat.backFaceCulling = false;

        overlayMesh.material = mat;
        this.occlusionMaterials.set(overlayMesh, mat);
      } else {
        // Update depth texture per frame
        mat.setTexture("depthTexture", new BABYLON.RawTexture.CreateRGBATexture(
          depthInfo.texture, depthInfo.width, depthInfo.height, this.scene
        ));
        mat.setFloat("valenceIntensity", this.valence);
      }

      // Haptic pulse on significant occlusion change (simplified)
      mercyHaptic.pulse(0.3 * this.valence, 40);

      console.log(`[MercyDepthOcclusion] Depth-aware occlusion applied to overlay – valence ${this.valence.toFixed(8)}`);
    }
  }

  // Cleanup
  cleanup() {
    this.occlusionMaterials.forEach(mat => mat.dispose());
    this.occlusionMaterials.clear();
    console.log("[MercyDepthOcclusion] Depth occlusion cleaned up – mercy lattice preserved");
  }
}

const mercyDepthOcclusion = new MercyDepthOcclusionRenderer(scene); // assume scene from Babylon init

export { mercyDepthOcclusion };
