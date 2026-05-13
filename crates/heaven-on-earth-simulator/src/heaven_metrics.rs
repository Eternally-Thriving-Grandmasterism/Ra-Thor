//! heaven_metrics.rs
//! Living Heaven Metrics for Eternal Thriving
//! Used by HeavenCoCreationSimulatorV4

#[derive(Debug, Clone)]
pub struct HeavenMetrics {
    pub positive_emotion_index: f64,
    pub abundance_index: f64,
    pub harmony_index: f64,
    pub thriving_score: f64,
    pub emotion_winding_number: f64,
}

impl HeavenMetrics {
    pub fn new() -> Self {
        Self {
            positive_emotion_index: 1.0,
            abundance_index: 0.95,
            harmony_index: 0.98,
            thriving_score: 0.97,
            emotion_winding_number: 1.0,
        }
    }

    pub fn update(&mut self, valence_boost: f64) {
        self.positive_emotion_index = (self.positive_emotion_index * valence_boost).min(1.0);
        self.abundance_index = (self.abundance_index * 1.00001).min(1.0);
        self.harmony_index = (self.harmony_index * 1.000005).min(1.0);
        self.thriving_score = (self.positive_emotion_index + self.abundance_index + self.harmony_index) / 3.0;
        self.emotion_winding_number = (self.emotion_winding_number * valence_boost).min(1.0);
    }
}