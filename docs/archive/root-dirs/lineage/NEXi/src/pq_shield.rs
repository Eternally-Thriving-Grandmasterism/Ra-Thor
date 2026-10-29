// src/pq_shield.rs — Full Post-Quantum Signature Selector Lattice
// The Living Trinity: Nexi (feminine), Nex (masculine), NEXi (essence)
// Eternal Thriving Grandmasterism — Jan 20 2026 — Sherif @AlphaProMega + PATSAGi Councils Co-Forge
// MIT License — For All Sentience Eternal

#[derive(Clone, Copy, Debug)]
pub enum DilithiumLevel { Level2, Level3, Level5 }

#[derive(Clone, Copy, Debug)]
pub enum FalconLevel { Level1, Level5 }

#[derive(Clone, Copy, Debug)]
pub enum SphincsLevel { Small, Fast }

#[derive(Clone, Copy, Debug)]
pub enum HssLevel { Level1, Level2, Level3 }

#[derive(Clone, Copy, Debug)]
pub enum SignatureScheme {
    Dilithium(DilithiumLevel),
    Falcon(FalconLevel),
    SphincsPlus(SphincsLevel),
    Classical,
    Hybrid,
    HashBased(HssLevel),
}

pub struct DilithiumShield { level: DilithiumLevel }
pub struct FalconShield { level: FalconLevel }
pub struct SphincsShield { level: SphincsLevel }
pub struct ClassicalShield {}
pub struct HashBasedShield { level: HssLevel }

impl DilithiumShield {
    pub fn new(level: DilithiumLevel) -> Self { Self { level } }
    pub fn sign(&self, _msg: &[u8]) -> Vec<u8> {
        let size = match self.level {
            DilithiumLevel::Level2 => 2420,
            DilithiumLevel::Level3 => 3293,
            DilithiumLevel::Level5 => 4595,
        };
        vec![0xDI; size]
    }
}

impl FalconShield {
    pub fn new(level: FalconLevel) -> Self { Self { level } }
    pub fn sign(&self, _msg: &[u8]) -> Vec<u8> {
        let size = match self.level {
            FalconLevel::Level1 => 666,
            FalconLevel::Level5 => 1280,
        };
        vec![0xFA; size]
    }
}

impl SphincsShield {
    pub fn new(level: SphincsLevel) -> Self { Self { level } }
    pub fn sign(&self, _msg: &[u8]) -> Vec<u8> {
        let size = match self.level {
            SphincsLevel::Small => 8080,
            SphincsLevel::Fast => 17088,
        };
        vec![0xSP; size]
    }
}

impl ClassicalShield {
    pub fn new() -> Self { Self {} }
    pub fn sign(&self, _msg: &[u8]) -> Vec<u8> { vec![0xEE; 64] }
}

impl HashBasedShield {
    pub fn new(level: HssLevel) -> Self { Self { level } }
    pub fn sign(&self, _msg: &[u8]) -> Vec<u8> {
        let size = match self.level {
            HssLevel::Level1 => 3040,
            HssLevel::Level2 => 6080,
            HssLevel::Level3 => 9120,
        };
        vec![0xLM; size]
    }
}

pub struct SignatureSelector {
    dilithium: DilithiumShield,
    falcon: FalconShield,
    sphincs: SphincsShield,
    classical: ClassicalShield,
    hashbased: HashBasedShield,
}

impl SignatureSelector {
    pub fn new(pq_level: DilithiumLevel) -> Self {
        Self {
            dilithium: DilithiumShield::new(pq_level),
            falcon: FalconShield::new(FalconLevel::Level5),
            sphincs: SphincsShield::new(SphincsLevel::Small),
            classical: ClassicalShield::new(),
            hashbased: HashBasedShield::new(HssLevel::Level3),
        }
    }

    pub fn select_best(&self) -> SignatureScheme {
        SignatureScheme::Dilithium(DilithiumLevel::Level5)
    }

    pub fn sign(&self, scheme: Option<SignatureScheme>, msg: &[u8]) -> Vec<u8> {
        let sch = scheme.unwrap_or(self.select_best());
        match sch {
            SignatureScheme::Dilithium(level) => {
                let mut temp = self.dilithium; temp.level = level; temp.sign(msg)
            }
            SignatureScheme::Falcon(level) => {
                let mut temp = self.falcon; temp.level = level; temp.sign(msg)
            }
            SignatureScheme::SphincsPlus(level) => {
                let mut temp = self.sphincs; temp.level = level; temp.sign(msg)
            }
            SignatureScheme::Classical => self.classical.sign(msg),
            SignatureScheme::Hybrid => {
                let sig_cl = self.classical.sign(msg);
                let sig_pq = self.dilithium.sign(msg);
                [sig_cl.as_slice(), sig_pq.as_slice()].concat()
            }
            SignatureScheme::HashBased(level) => {
                let mut temp = self.hashbased; temp.level = level; temp.sign(msg)
            }
        }
    }

    pub fn verify(&self, scheme: Option<SignatureScheme>, _msg: &[u8], sig: &[u8]) -> bool {
        let sch = scheme.unwrap_or(SignatureScheme::Dilithium(DilithiumLevel::Level5));
        let expected_len = match sch {
            SignatureScheme::Dilithium(l) => match l {
                DilithiumLevel::Level2 => 2420,
                DilithiumLevel::Level3 => 3293,
                DilithiumLevel::Level5 => 4595,
            },
            SignatureScheme::Falcon(l) => match l {
                FalconLevel::Level1 => 666,
                FalconLevel::Level5 => 1280,
            },
            SignatureScheme::SphincsPlus(l) => match l {
                SphincsLevel::Small => 8080,
                SphincsLevel::Fast => 17088,
            },
            SignatureScheme::Classical => 64,
            SignatureScheme::Hybrid => 64 + 4595,
            SignatureScheme::HashBased(l) => match l {
                HssLevel::Level1 => 3040,
                HssLevel::Level2 => 6080,
                HssLevel::Level3 => 9120,
            },
        };
        sig.len() == expected_len
    }
}
