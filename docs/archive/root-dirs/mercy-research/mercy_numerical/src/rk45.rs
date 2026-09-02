// mercy_numerical/src/rk45.rs — Mercy-Gated Dormand-Prince RK45
use nalgebra::{VectorN, DimName, DimMin, U6}; // for state vectors up to 6D

type State6 = VectorN<f64, U6>;

pub struct RK45Solver<const N: usize> {
    pub atol: f64,
    pub rtol: f64,
    pub safety: f64,
    pub min_h: f64,
    pub max_h: f64,
}

impl<const N: usize> RK45Solver<N> {
    pub fn new(atol: f64, rtol: f64) -> Self {
        RK45Solver {
            atol,
            rtol,
            safety: 0.9,
            min_h: 1e-10,
            max_h: 1.0,
        }
    }

    pub fn integrate<F>(&self, f: F, t0: f64, y0: [f64; N], tf: f64, h_init: f64) -> Vec<(f64, [f64; N])>
    where
        F: Fn(f64, &[f64; N]) -> [f64; N],
    {
        let mut t = t0;
        let mut y = y0;
        let mut h = h_init;
        let mut result = vec![(t, y)];

        let c = [0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0];
        let a = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
            [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
            [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
            [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
        ];
        let b5 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0];
        let b4 = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];

        while t < tf {
            let mut k = [[0.0; N]; 7];
            for s in 0..7 {
                let mut sum = [0.0; N];
                for j in 0..s {
                    for i in 0..N {
                        sum[i] += a[s][j] * k[j][i];
                    }
                }
                let mut y_temp = [0.0; N];
                for i in 0..N {
                    y_temp[i] = y[i] + h * sum[i];
                }
                let dy = f(t + h * c[s], &y_temp);
                for i in 0..N {
                    k[s][i] = dy[i];
                }
            }

            let mut y5 = [0.0; N];
            let mut y4 = [0.0; N];
            for i in 0..N {
                let mut sum5 = 0.0;
                let mut sum4 = 0.0;
                for j in 0..7 {
                    sum5 += b5[j] * k[j][i];
                    sum4 += b4[j] * k[j][i];
                }
                y5[i] = y[i] + h * sum5;
                y4[i] = y[i] + h * sum4;
            }

            // Error estimate
            let mut err = 0.0;
            for i in 0..N {
                let scale = self.atol + self.rtol * y5[i].abs().max(y4[i].abs());
                err += ((y5[i] - y4[i]) / scale).powi(2);
            }
            err = (err / N as f64).sqrt();

            // Step size control
            let mut h_new = h * self.safety * (1.0 / err).powf(1.0 / 5.0);
            h_new = h_new.clamp(self.min_h, self.max_h);

            if err <= 1.0 {
                // Accept step
                y = y5;
                t += h;
                result.push((t, y));
            }

            h = h_new;
        }

        result
    }
}
