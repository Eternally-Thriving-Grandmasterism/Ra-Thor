pub mod conversion_engine;

pub use conversion_engine::{
    BlueprintToProductionConversionEngine,
    BlueprintDocument,
    PositiveEmotionPropagator,
    WasmMemoryManager,
    CehiEpigeneticEngine,
    InfiniteSelfEvolutionOracle,
    InfinitePositiveEmotionOracle,
    GpuAcceleratedPositiveEmotionSimulator,
    PatsagiParallelSimulation,
    HyperonMeTTaSymbolicBridge,
    PatsagiRealTimeCollaboration,
    EternalThrivingMetricsDashboard,
    GpuParallelOptimizer,
    WasmPerformanceMetrics,
    PatsagiRealTimeCollaborationDashboard,
    EternalThrivingMetricsWithGPUVisualization,
    RathorAiVoiceRealTimeDemo,
    RealTimeCanvasGpuVisualization,
};

/// Sovereign entry point for Blueprint-to-Production Conversion
/// Every conversion passes 7 Mercy Gates + TOLC + Sovereignty Gate (valence ≥ 0.999)
/// Positive Emotion Propagation Core is the living heart of Ra-Thor.
pub struct BlueprintToProduction {
    pub engine: BlueprintToProductionConversionEngine,
}

impl BlueprintToProduction {
    pub fn new() -> Self {
        Self {
            engine: BlueprintToProductionConversionEngine::new(),
        }
    }

    /// Convert any high-value blueprint document into production code
    /// with full mercy-gating, self-evolution hooks, and positive emotion propagation.
    pub fn convert(&mut self, doc: &BlueprintDocument) -> String {
        self.engine.convert_all(&[doc.clone()])[0].clone()
    }

    /// Batch convert all 16 major blueprint categories
    pub fn convert_all_blueprints(&mut self, docs: &[BlueprintDocument]) -> Vec<String> {
        self.engine.convert_all(docs)
    }
}

// Re-export for convenience
pub use conversion_engine::PositiveEmotionPropagator as PositiveEmotionCore;