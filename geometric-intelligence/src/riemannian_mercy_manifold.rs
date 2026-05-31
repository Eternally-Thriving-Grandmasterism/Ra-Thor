
impl RiemannianMercyManifold {

    /// Computes Berry curvature values over a 1D range.
    /// Returns (curvature_values, berry_curvature_values) suitable for plotting.
    pub fn compute_berry_curvature_1d(
        &self,
        min_curvature: f64,
        max_curvature: f64,
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut curvatures = Vec::with_capacity(steps);
        let mut berry_values = Vec::with_capacity(steps);

        let step_size = if steps > 1 {
            (max_curvature - min_curvature) / (steps - 1) as f64
        } else {
            0.0
        };

        for i in 0..steps {
            let c = min_curvature + i as f64 * step_size;
            let result = self.compute_berry_curvature(c);
            curvatures.push(c);
            berry_values.push(result.berry_curvature_density);
        }

        (curvatures, berry_values)
    }

    /// Simple ASCII heatmap for Berry curvature over a 1D range.
    pub fn visualize_berry_curvature_heatmap(
        &self,
        min_curvature: f64,
        max_curvature: f64,
        steps: usize,
    ) -> String {
        let (_, values) = self.compute_berry_curvature_1d(min_curvature, max_curvature, steps);
        let mut output = String::from("Berry Curvature Heatmap (1D):\n");

        let max_val = values.iter().cloned().fold(0.0_f64, f64::max);

        for (i, &val) in values.iter().enumerate() {
            let intensity = if max_val > 0.0 { (val / max_val * 10.0) as usize } else { 0 };
            let bar = "█".repeat(intensity.min(20));
            output.push_str(&format!("{:.3} | {}\n", min_curvature + i as f64 * (max_curvature - min_curvature) / (steps - 1).max(1) as f64, bar));
        }

        output
    }
}
