
impl RiemannianMercyManifold {

    /// Returns the cumulative Berry phase evolution over a sequence.
    /// Useful for plotting phase vs step.
    pub fn compute_berry_phase_evolution(
        &self,
        curvatures: &[f64],
        areas: &[f64],
    ) -> Vec<f64> {
        let mut cumulative = 0.0;
        let mut evolution = Vec::with_capacity(curvatures.len());

        for (curv, area) in curvatures.iter().zip(areas.iter()) {
            cumulative += self.estimate_holonomy(*curv, *area);
            evolution.push(cumulative);
        }

        evolution
    }

    /// Generates a simple textual visualization of Berry phase accumulation.
    pub fn visualize_berry_phase_text(
        &self,
        curvatures: &[f64],
        areas: &[f64],
    ) -> String {
        let evolution = self.compute_berry_phase_evolution(curvatures, areas);
        let mut output = String::from("Berry Phase Evolution:\n");

        for (i, phase) in evolution.iter().enumerate() {
            let bar_length = ((phase.abs() * 10.0) as usize).min(40);
            let bar = if *phase >= 0.0 {
                "█".repeat(bar_length)
            } else {
                "▓".repeat(bar_length)
            };
            output.push_str(&format!("Step {:>2}: {:>6.3} | {}\n", i, phase, bar));
        }

        output
    }

    /// Pretty print Berry Phase + Curvature summary
    pub fn print_berry_summary(&self, curvatures: &[f64], areas: &[f64]) {
        let phase_result = self.compute_berry_phase_analog(curvatures, areas);
        let curvature_result = if !curvatures.is_empty() {
            Some(self.compute_berry_curvature(curvatures[0]))
        } else {
            None
        };

        println!("=== Berry Phase Summary ===");
        println!("Final Phase: {:.4}", phase_result.phase);
        println!("Magnitude  : {:.4}", phase_result.magnitude);
        println!("Interpretation: {}", phase_result.interpretation);

        if let Some(curv) = curvature_result {
            println!("Initial Berry Curvature: {:.4} (effective {:.4})", curv.raw_curvature, curv.effective_curvature);
        }
    }
}
