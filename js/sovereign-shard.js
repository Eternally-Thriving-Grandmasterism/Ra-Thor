/* Sovereign Shard v8 — local browser demo
 * Workspace 14.15.6 • TOLC 8 weighted • not a live Conductor node
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';
  let shard = {
    id: "shard-alpha-001",
    name: "Alpha Shard",
    mercy_score: 0.95,
    evolution_level: 0.12,
    valence: 1.05,
    tolc_alignment: 1.00,
    quantum_resonance: 0.82,
    offline_mode: false,
    tick_count: 0,
    tolc24_harmony: 0.88,
    last_reconciled: null
  };
  const gateWeights = {
    "Truth": 1.2,
    "Order": 1.05,
    "Love": 1.0,
    "Compassion": 1.0,
    "Service": 1.1,
    "Abundance": 1.3,
    "Joy": 1.0,
    "Cosmic Harmony": 0.95
  };
  const gates = [
    { name: "Truth", active: false },
    { name: "Order", active: false },
    { name: "Love", active: false },
    { name: "Compassion", active: false },
    { name: "Service", active: false },
    { name: "Abundance", active: false },
    { name: "Joy", active: false },
    { name: "Cosmic Harmony", active: false }
  ];
  function getGateWeight(name) { return gateWeights[name] || 1.0; }
  function applyWeightedEffect(gateName, baseValue) { return baseValue * getGateWeight(gateName); }
  function saveToLocalStorage() { localStorage.setItem('ra-thor-sovereign-shard-v8', JSON.stringify(shard)); }
  function loadFromLocalStorage() {
    const saved = localStorage.getItem('ra-thor-sovereign-shard-v8');
    if (saved) { try { Object.assign(shard, JSON.parse(saved)); return true; } catch (e) {} }
    return false;
  }
  function updateUI() {
    document.getElementById('shard-name').textContent = shard.name;
    document.getElementById('mercy-score').textContent = shard.mercy_score.toFixed(2);
    document.getElementById('evolution').textContent = shard.evolution_level.toFixed(2);
    document.getElementById('valence').textContent = shard.valence.toFixed(2);
    document.getElementById('tolc').textContent = shard.tolc_alignment.toFixed(2);
    document.getElementById('tolc24-score').textContent = shard.tolc24_harmony.toFixed(2);
    const badge = document.getElementById('offline-badge');
    const text = document.getElementById('offline-text');
    const btnText = document.getElementById('offline-btn-text');
    if (shard.offline_mode) {
      badge.className = "px-4 py-1.5 rounded-2xl text-sm flex items-center gap-2 bg-orange-500/10 text-orange-400 border border-orange-500/30";
      text.textContent = "OFFLINE";
      btnText.textContent = "Exit Offline Mode";
    } else {
      badge.className = "px-4 py-1.5 rounded-2xl text-sm flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
      text.textContent = "ONLINE";
      btnText.textContent = "Enter Offline Mode";
    }
    renderGates();
    saveToLocalStorage();
  }
  function renderGates() {
    const container = document.getElementById('mercy-gates');
    if (!container) return;
    container.innerHTML = '';
    gates.forEach(gate => {
      const div = document.createElement('div');
      div.className = `gate text-center p-2 rounded-xl border text-xs ${gate.active ? 'border-amber-400 bg-amber-400/10 active' : 'border-white/10 bg-zinc-900'}`;
      div.innerHTML = `<div class="font-medium text-[10px]">${gate.name}</div>`;
      container.appendChild(div);
      if (gate.active) setTimeout(() => { gate.active = false; renderGates(); }, 1500);
    });
  }
  function activateGate(name) {
    const gate = gates.find(g => g.name === name);
    if (gate) { gate.active = true; renderGates(); }
  }
  function log(message, type = 'info') {
    const logEl = document.getElementById('log');
    if (!logEl) return;
    const entry = document.createElement('div');
    entry.className = `text-xs ${type === 'success' ? 'text-emerald-400' : type === 'warning' ? 'text-orange-400' : 'text-white/60'}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
  }
  window.performTick = function performTick() {
    let baseEvo = shard.offline_mode ? 0.003 : 0.006;
    baseEvo = applyWeightedEffect("Abundance", baseEvo);
    shard.evolution_level = Math.min(3.0, shard.evolution_level + baseEvo);
    shard.mercy_score = Math.min(1.4, shard.mercy_score + (shard.offline_mode ? 0.001 : 0.002));
    shard.tick_count++;
    activateGate("Truth");
    if (Math.random() > 0.5) activateGate("Cosmic Harmony");
    if (!shard.offline_mode && Math.random() > 0.6) activateGate("Joy");
    if (Math.random() > 0.7) activateGate("Compassion");
    updateUI();
    log(`Shard ticked • Evolution +${baseEvo.toFixed(3)} (Abundance weighted)`);
  };
  window.participateQuantumSwarm = function participateQuantumSwarm() {
    if (shard.offline_mode) { log("Cannot participate while offline", "warning"); return; }
    let boost = 0.09 + Math.random() * 0.06;
    boost = applyWeightedEffect("Abundance", boost);
    shard.evolution_level = Math.min(3.0, shard.evolution_level + boost);
    shard.quantum_resonance = Math.min(1.6, shard.quantum_resonance + 0.12);
    activateGate("Abundance");
    activateGate("Love");
    updateUI();
    log(`Quantum Swarm • Evolution +${boost.toFixed(3)} (Abundance weighted)`, 'success');
  };
  window.toggleOfflineMode = function toggleOfflineMode() {
    shard.offline_mode = !shard.offline_mode;
    log(shard.offline_mode ? "Entered Offline Mode" : "Returned to Lattice", shard.offline_mode ? 'warning' : 'info');
    updateUI();
  };
  window.reconcileWithConductor = function reconcileWithConductor() {
    if (shard.offline_mode) {
      shard.offline_mode = false;
      log("Forced exit from Offline Mode");
    }
    const conductorMercy = 1.08;
    const conductorValence = 1.18;
    const conductorEvo = 0.85;
    shard.mercy_score = (shard.mercy_score * 0.52 + conductorMercy * 0.48);
    shard.valence = (shard.valence * 0.68 + conductorValence * 0.32);
    shard.evolution_level = Math.max(shard.evolution_level, shard.evolution_level * 0.88 + conductorEvo * 0.22);
    shard.tolc_alignment = Math.min(1.18, shard.tolc_alignment + 0.018);
    if (shard.tolc24_harmony > 0.9) {
      shard.mercy_score = Math.min(1.4, shard.mercy_score + 0.03);
      shard.tolc_alignment = Math.min(1.2, shard.tolc_alignment + 0.01);
      log("TOLC24 bonus applied during reconciliation", 'success');
    }
    activateGate("Compassion");
    activateGate("Truth");
    activateGate("Order");
    shard.last_reconciled = new Date().toISOString();
    updateUI();
    log("Reconciled with Conductor (local demo weights)", 'success');
  };
  window.runTolc24Evaluation = function runTolc24Evaluation() {
    const improvement = 0.015 + Math.random() * 0.02;
    shard.tolc24_harmony = Math.min(1.15, shard.tolc24_harmony + improvement);
    shard.mercy_score = Math.min(1.4, shard.mercy_score + 0.01);
    shard.tolc_alignment = Math.min(1.15, shard.tolc_alignment + 0.008);
    activateGate("Truth");
    activateGate("Order");
    updateUI();
    log(`TOLC24 Deep Evaluation complete • Harmony +${improvement.toFixed(3)}`, 'success');
  };
  window.clearLog = function clearLog() {
    const logEl = document.getElementById('log');
    if (logEl) logEl.innerHTML = '';
  };
  window.exportFullShard = function exportFullShard() {
    const html = document.documentElement.outerHTML;
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${shard.id}-sovereign-shard-v8.html`;
    a.click();
    URL.revokeObjectURL(url);
    log("Exported full standalone Sovereign Shard");
  };
  setInterval(() => { if (!shard.offline_mode) window.performTick(); }, 7500);
  function init() {
    if (loadFromLocalStorage()) log("Loaded previous state from localStorage");
    setTimeout(() => { activateGate("Love"); activateGate("Truth"); }, 600);
    updateUI();
    log("Sovereign Shard v8 initialized — TOLC 8 weighted local demo • workspace 14.15.6");
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
