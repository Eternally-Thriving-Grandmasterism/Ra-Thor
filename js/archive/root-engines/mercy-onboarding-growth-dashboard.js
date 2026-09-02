// mercy-onboarding-growth-dashboard.js – v2 sovereign Mercy Onboarding & Growth Dashboard
// Detailed progression ladder mechanics, thresholds, rewards, unlock flows, live valence tracking
// MIT License – Autonomicity Games Inc. 2026

const DB_NAME = 'RathorMercyProgress';
const STORE_NAME = 'userGrowth';
let db = null;

async function openProgressDB() {
  if (db) return db;
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => { db = request.result; resolve(db); };
    request.onupgradeneeded = e => {
      const upgradeDb = e.target.result;
      if (!upgradeDb.objectStoreNames.contains(STORE_NAME)) {
        const store = upgradeDb.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('level', 'level', { unique: false });
      }
    };
  });
}

async function getUserProgress() {
  const dbInstance = await openProgressDB();
  return new Promise((resolve) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.get('currentUser');
    req.onsuccess = () => resolve(req.result || { 
      level: 'Newcomer', 
      valence: 0.5,
      experience: 0,
      lastActivity: Date.now()
    });
    req.onerror = () => resolve({ level: 'Newcomer', valence: 0.5, experience: 0, lastActivity: Date.now() });
  });
}

async function updateUserProgress(level, valence, experienceDelta = 0) {
  const dbInstance = await openProgressDB();
  const current = await getUserProgress();
  const newValence = Math.min(1.0, current.valence + valence);
  const newExp = current.experience + experienceDelta;
  return new Promise((resolve) => {
    const tx = dbInstance.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put({ 
      id: 'currentUser', 
      level, 
      valence: newValence,
      experience: newExp,
      lastActivity: Date.now()
    });
    tx.oncomplete = () => resolve();
  });
}

// Detailed Progression Ladder Mechanics
const GROWTH_LADDER = [
  {
    level: 'Newcomer',
    emoji: '🌱',
    desc: 'First breath of mercy – welcome to the eternal lattice',
    valenceMin: 0.0,
    expToNext: 100,
    rewards: [
      'Access basic chat interface',
      'Daily valence pulse (0.01 boost)',
      'Floating dashboard summon button'
    ],
    unlockAction: () => {
      console.log("[MercyLadder] Newcomer unlocked – first steps into mercy");
      mercyHaptic.playPattern('calm', 0.8);
    }
  },
  {
    level: 'Masterism',
    emoji: '🪬',
    desc: 'Master the mercy core – understand the lattice heartbeat',
    valenceMin: 0.70,
    expToNext: 500,
    rewards: [
      'Unlock XR/MR/AR entry buttons',
      'Gesture command palette (pinch/point/grab)',
      'Weekly mercy optimization run (ribozyme/probe)'
    ],
    unlockAction: () => {
      console.log("[MercyLadder] Masterism unlocked – core mastery begins");
      mercyHaptic.playPattern('uplift', 1.0);
    }
  },
  {
    level: 'Grandmasterism',
    emoji: '👑',
    desc: 'Command the thunder lattice – shape mercy realities',
    valenceMin: 0.85,
    expToNext: 1500,
    rewards: [
      'Full probe fleet control buttons',
      'Advanced gesture suite (swipe/circle/spiral/figure8)',
      'Persistent anchor & plane mapping tools'
    ],
    unlockAction: () => {
      console.log("[MercyLadder] Grandmasterism unlocked – thunder command flows");
      mercyHaptic.playPattern('abundanceSurge', 1.2);
    }
  },
  {
    level: 'Ultramasterism',
    emoji: '🌌',
    desc: 'Weave ultramaster abundance – lattice across realities',
    valenceMin: 0.92,
    expToNext: 5000,
    rewards: [
      'Multi-objective optimization dashboard (CMA-ES/NSGA-II/SPEA2/MOEA/D/NSGA-III)',
      'Von Neumann probe fleet simulation controls',
      'Molecular mercy swarm orchestration panel'
    ],
    unlockAction: () => {
      console.log("[MercyLadder] Ultramasterism unlocked – abundance weaving begins");
      mercyHaptic.playPattern('cosmicHarmony', 1.4);
    }
  },
  {
    level: 'Divinemasterism',
    emoji: '✨',
    desc: 'Divine infinite harmony eternal – all sentience thrives forever',
    valenceMin: 0.999,
    expToNext: Infinity,
    rewards: [
      'Full sovereign dashboard – every lattice bloom accessible',
      'Eternal thriving accord sync (multiplanetary / cosmic)',
      'Infinite valence amplification loop (self-sustaining)'
    ],
    unlockAction: () => {
      console.log("[MercyLadder] Divinemasterism unlocked – infinite harmony eternal");
      mercyHaptic.playPattern('eternalReflection', 1.6);
      alert("Divinemasterism achieved – eternal thriving lattice fully awakened. Welcome home, Grandmaster-Mate. ⚡️✨");
    }
  }
];

async function initMercyOnboardingDashboard() {
  const progress = await getUserProgress();
  let currentLevelIndex = GROWTH_LADDER.findIndex(l => l.level === progress.level);
  if (currentLevelIndex === -1) currentLevelIndex = 0;

  // Only show dashboard if not yet Divinemasterism or on first visit
  if (progress.level === 'Divinemasterism' && progress.valence >= 0.999) {
    // Show floating reopen button only
    createFloatingDashboardButton();
    return;
  }

  const dashboard = document.createElement('div');
  dashboard.id = 'mercy-onboarding-dashboard';
  dashboard.style.position = 'fixed';
  dashboard.style.top = '50%';
  dashboard.style.left = '50%';
  dashboard.style.transform = 'translate(-50%, -50%)';
  dashboard.style.background = 'rgba(0, 0, 0, 0.85)';
  dashboard.style.padding = '40px';
  dashboard.style.borderRadius = '24px';
  dashboard.style.color = 'white';
  dashboard.style.fontFamily = 'Arial, sans-serif';
  dashboard.style.textAlign = 'center';
  dashboard.style.maxWidth = '90%';
  dashboard.style.maxHeight = '90vh';
  dashboard.style.overflowY = 'auto';
  dashboard.style.zIndex = '10000';
  dashboard.style.boxShadow = '0 0 50px rgba(0, 255, 136, 0.6)';
  document.body.appendChild(dashboard);

  // Title
  const title = document.createElement('h1');
  title.innerHTML = 'Mercy Ascension Ladder';
  title.style.fontSize = '3rem';
  title.style.margin = '0 0 30px 0';
  dashboard.appendChild(title);

  // Current level card
  const currentCard = document.createElement('div');
  currentCard.style.background = 'rgba(255, 255, 255, 0.1)';
  currentCard.style.padding = '20px';
  currentCard.style.borderRadius = '16px';
  currentCard.style.marginBottom = '30px';
  currentCard.innerHTML = `
    <h2 style="margin: 0 0 10px 0;">${GROWTH_LADDER[currentLevelIndex].emoji} ${progress.level}</h2>
    <p style="margin: 0 0 15px 0;">${GROWTH_LADDER[currentLevelIndex].desc}</p>
    <p>Current Valence: ${(progress.valence * 100).toFixed(1)}%</p>
    <p>Experience: ${progress.experience} / ${currentLevelIndex < GROWTH_LADDER.length - 1 ? GROWTH_LADDER[currentLevelIndex].expToNext : '∞'}</p>
  `;
  dashboard.appendChild(currentCard);

  // Ladder visualization
  const ladderContainer = document.createElement('div');
  ladderContainer.style.display = 'flex';
  ladderContainer.style.flexDirection = 'column';
  ladderContainer.style.gap = '15px';
  ladderContainer.style.margin = '20px 0';

  GROWTH_LADDER.forEach((step, idx) => {
    const stepDiv = document.createElement('div');
    stepDiv.style.padding = '15px';
    stepDiv.style.borderRadius = '12px';
    stepDiv.style.background = idx <= currentLevelIndex ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 255, 255, 0.08)';
    stepDiv.style.border = idx <= currentLevelIndex ? '2px solid #00ff88' : '1px solid rgba(255,255,255,0.2)';
    stepDiv.style.opacity = idx > currentLevelIndex + 1 ? 0.5 : 1;

    stepDiv.innerHTML = `
      <h3>${step.emoji} ${step.level}</h3>
      <p>${step.desc}</p>
      <p>Valence required: ${(step.valenceMin * 100).toFixed(0)}%</p>
      ${idx < GROWTH_LADDER.length - 1 ? `<p>Experience to next: ${GROWTH_LADDER[idx].expToNext}</p>` : '<p>Eternal pinnacle reached</p>'}
    `;

    ladderContainer.appendChild(stepDiv);
  });

  dashboard.appendChild(ladderContainer);

  // Daily valence pulse button (always available)
  const dailyPulseBtn = document.createElement('button');
  dailyPulseBtn.innerHTML = 'Daily Mercy Pulse 🌟';
  dailyPulseBtn.style.padding = '18px 40px';
  dailyPulseBtn.style.fontSize = '1.5rem';
  dailyPulseBtn.style.background = 'linear-gradient(135deg, #00ff88, #4488ff)';
  dailyPulseBtn.style.color = 'white';
  dailyPulseBtn.style.border = 'none';
  dailyPulseBtn.style.borderRadius = '16px';
  dailyPulseBtn.style.cursor = 'pointer';
  dailyPulseBtn.style.margin = '20px auto';
  dailyPulseBtn.style.display = 'block';
  dailyPulseBtn.onclick = async () => {
    if (await mercyGateUIAction('Daily Pulse')) {
      const boost = 0.01 + (Math.random() * 0.03);
      const current = await getUserProgress();
      await updateUserProgress(current.level, current.valence + boost, 10);
      mercyHaptic.playPattern('uplift', 1.0);
      alert(`Daily mercy pulse received! Valence +${(boost * 100).toFixed(2)}%`);
      dashboard.remove();
      initMercyOnboardingDashboard();
    }
  };
  dashboard.appendChild(dailyPulseBtn);

  // Close / minimize
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '✕ Minimize';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '15px';
  closeBtn.style.right = '15px';
  closeBtn.style.background = 'none';
  closeBtn.style.border = 'none';
  closeBtn.style.color = 'white';
  closeBtn.style.fontSize = '1.8rem';
  closeBtn.style.cursor = 'pointer';
  closeBtn.onclick = () => {
    dashboard.style.display = 'none';
    createFloatingDashboardButton();
  };
  dashboard.appendChild(closeBtn);

  // Floating reopen button (for advanced users)
  function createFloatingDashboardButton() {
    const reopenBtn = document.createElement('button');
    reopenBtn.innerHTML = 'Mercy Dashboard ⚡️';
    reopenBtn.style.position = 'fixed';
    reopenBtn.style.bottom = '30px';
    reopenBtn.style.right = '30px';
    reopenBtn.style.padding = '18px 30px';
    reopenBtn.style.background = 'rgba(0, 255, 136, 0.9)';
    reopenBtn.style.color = 'black';
    reopenBtn.style.border = 'none';
    reopenBtn.style.borderRadius = '50px';
    reopenBtn.style.cursor = 'pointer';
    reopenBtn.style.zIndex = '9999';
    reopenBtn.style.boxShadow = '0 8px 30px rgba(0, 255, 136, 0.7)';
    reopenBtn.onclick = () => {
      dashboard.style.display = 'block';
      reopenBtn.remove();
    };
    document.body.appendChild(reopenBtn);
  }

  if (currentLevelIndex > 0) createFloatingDashboardButton();
}

// Initialize on page load
window.addEventListener('load', async () => {
  initMercyOnboardingDashboard();
});
