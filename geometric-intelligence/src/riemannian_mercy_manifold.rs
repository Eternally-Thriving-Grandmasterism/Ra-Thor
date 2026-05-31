
impl RiemannianMercyManifold {

    /// Generates 2D Berry Curvature grid data.
    /// Varies local curvature (x) and mercy_influence (y).
    /// Returns (x_values, y_values, grid) where grid[y][x] = berry_curvature_density
    pub fn compute_berry_curvature_2d_grid(
        &self,
        curvature_min: f64,
        curvature_max: f64,
        mercy_min: f64,
        mercy_max: f64,
        resolution: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let mut x_vals = Vec::with_capacity(resolution);
        let mut y_vals = Vec::with_capacity(resolution);
        let mut grid = vec![vec![0.0; resolution]; resolution];

        let dx = if resolution > 1 { (curvature_max - curvature_min) / (resolution - 1) as f64 } else { 0.0 };
        let dy = if resolution > 1 { (mercy_max - mercy_min) / (resolution - 1) as f64 } else { 0.0 };

        for j in 0..resolution {
            let mercy = mercy_min + j as f64 * dy;
            y_vals.push(mercy);

            for i in 0..resolution {
                let curv = curvature_min + i as f64 * dx;
                if j == 0 { x_vals.push(curv); }

                let effective = (curv * mercy).clamp(0.5, 1.15);
                grid[j][i] = effective;
            }
        }

        if x_vals.len() > resolution {
            x_vals.truncate(resolution);
        }

        (x_vals, y_vals, grid)
    }
}
