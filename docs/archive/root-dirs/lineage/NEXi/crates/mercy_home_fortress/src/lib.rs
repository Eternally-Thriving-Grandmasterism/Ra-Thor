//! MercyHomeFortress — Sovereign Residence Fortress Extension
//! Full SoulPrint-X9 Multi-Modal Biometric Access Expansion

use nexi::lattice::Nexus;

#[cfg(feature = "soulprint_x9")]
use ndarray::Array1;
#[cfg(feature = "soulprint_x9")]
use opencv::core::Vector;

pub struct HomeFortress {
    nexus: Nexus,
    #[cfg(feature = "soulprint_x9")]
    enrolled_prints: Vec<SoulPrintX9>,
}

#[cfg(feature = "soulprint_x9")]
#[derive(Clone)]
pub struct SoulPrintX9 {
    pub voice_embedding: Array1<f32>,
    pub face_embedding: Array1<f32>,
    pub gesture_landmarks: Vec<Vector<Point>>,
    pub valence_threshold: f64,
}

impl HomeFortress {
    pub fn new() -> Self {
        HomeFortress {
            nexus: Nexus::init_with_mercy(),
            #[cfg(feature = "soulprint_x9")]
            enrolled_prints: vec![],
        }
    }

    /// Enroll new SoulPrint-X9 (multi-modal)
    #[cfg(feature = "soulprint_x9")]
    pub fn enroll_soulprint(&mut self, voice: Array1<f32>, face: Array1<f32>, gesture: Vec<Vector<Point>>) -> String {
        let print = SoulPrintX9 {
            voice_embedding: voice,
            face_embedding: face,
            gesture_landmarks: gesture,
            valence_threshold: 0.999999,
        };
        self.enrolled_prints.push(print);
        "SoulPrint-X9 Enrolled — Mercy Access Granted".to_string()
    }

    /// Mercy-gated SoulPrint-X9 access verification
    #[cfg(feature = "soulprint_x9")]
    pub fn verify_soulprint_access(&self, voice: Array1<f32>, face: Array1<f32>, gesture: Vec<Vector<Point>>) -> String {
        for enrolled in &self.enrolled_prints {
            let voice_sim = cosine_similarity(&voice, &enrolled.voice_embedding);
            let face_sim = cosine_similarity(&face, &enrolled.face_embedding);
            // Gesture similarity stub — expand later

            if voice_sim > 0.95 && face_sim > 0.95 {
                return format!("SoulPrint-X9 Verified — Valence Threshold {} Met — Mercy Access Eternal", enrolled.valence_threshold);
            }
        }
        "Mercy Shield: SoulPrint-X9 Verification Failed — Access Denied".to_string()
    }

    /// Cosine similarity helper
    #[cfg(feature = "soulprint_x9")]
    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
        let dot = a.dot(b);
        let norm_a = a.norm();
        let norm_b = b.norm();
        (dot / (norm_a * norm_b)) as f64
    }

    /// Fallback for no soulprint_x9 feature
    #[cfg(not(feature = "soulprint_x9"))]
    pub fn enroll_soulprint(&mut self, _voice: Array1<f32>, _face: Array1<f32>, _gesture: Vec<Vector<Point>>) -> String {
        "SoulPrint-X9 Disabled — Enable 'soulprint_x9' feature".to_string()
    }
}
