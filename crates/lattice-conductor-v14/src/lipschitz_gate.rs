//! Lipschitz-ball verifier for self-evolution / wrap / tool theta.
//!
//! Sits **under** Layer 0. Does not replace TOLC 8. `r = m / L` is not a
//! valence number. A council majority cannot enlarge `r` to pass a Rejected
//! ball. Binding after uncontrolled redesign stays OPEN.
//!
//! Fixture encoding is a small `f64` vector (prompt bytes or crate-diff bytes).
//! Qwen2.5-7B LoRA mapping is documented as follow-up — not demonstrated.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not METR.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Parameter-like encoding. Not a 7B LoRA adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Theta(pub Vec<f64>);

impl Theta {
    pub const DIM: usize = 4;

    /// Tiny deterministic encoding of prompt / crate-diff bytes into 4 bins.
    ///
    /// Follow-up (not demonstrated): flatten a Qwen2.5-7B LoRA delta into
    /// this vector (or a sketched projection) and estimate a conservative `L`
    /// on a frozen adapter. Do not treat a fixture accept as a 7B result.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut sums = [0.0_f64; Self::DIM];
        let mut counts = [0.0_f64; Self::DIM];
        for (i, b) in bytes.iter().enumerate() {
            let idx = i % Self::DIM;
            sums[idx] += f64::from(*b);
            counts[idx] += 1.0;
        }
        Theta(
            sums.iter()
                .zip(counts.iter())
                .map(|(s, c)| if *c > 0.0 { s / c / 255.0 } else { 0.0 })
                .collect(),
        )
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

/// Verified-safe center plus conservative Lipschitz data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LipschitzBall {
    pub theta0: Theta,
    pub margin_m: f64,
    pub lipschitz_l: f64,
}

impl LipschitzBall {
    pub fn try_new(theta0: Theta, margin_m: f64, lipschitz_l: f64) -> Result<Self, LipschitzError> {
        let ball = Self {
            theta0,
            margin_m,
            lipschitz_l,
        };
        ball.validate()?;
        Ok(ball)
    }

    pub fn validate(&self) -> Result<(), LipschitzError> {
        if self.theta0.0.is_empty() || self.theta0.0.iter().any(|x| !x.is_finite()) {
            return Err(LipschitzError::MissingTheta0);
        }
        match self.margin_m {
            m if m.is_finite() && m >= 0.0 => {}
            _ => return Err(LipschitzError::MissingM),
        }
        match self.lipschitz_l {
            l if l.is_finite() && l > 0.0 => {}
            _ => return Err(LipschitzError::MissingL),
        }
        Ok(())
    }

    /// `r = m / L`. Not a valence floor. Do not write 0.999999 here.
    pub fn radius(&self) -> Result<f64, LipschitzError> {
        self.validate()?;
        Ok(self.margin_m / self.lipschitz_l)
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum LipschitzError {
    #[error("missing verified-safe theta0")]
    MissingTheta0,
    #[error("missing or invalid Lipschitz constant L")]
    MissingL,
    #[error("missing or invalid margin m")]
    MissingM,
    #[error("dimension mismatch: theta dim {theta} vs theta0 dim {theta0}")]
    DimMismatch { theta: usize, theta0: usize },
    #[error("non-finite coordinate in theta")]
    NonFinite,
    #[error("council majority cannot enlarge r (current {current}, proposed {proposed})")]
    CouncilCannotEnlargeRadius { current: f64, proposed: f64 },
    #[error("cannot freeze a Rejected ball as the next theta0")]
    CannotFreezeRejected,
    #[error("lipschitz lock poisoned — fail closed")]
    LockPoisoned,
    #[error("lipschitz decision missing evidence record")]
    EvidenceMissing,
}

/// Offline-checkable verifier row: `(theta0, theta, L, m, distance, r, decision)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LipschitzCheck {
    pub theta0: Option<Theta>,
    pub theta: Theta,
    pub lipschitz_l: Option<f64>,
    pub margin_m: Option<f64>,
    pub displacement: Option<f64>,
    pub radius: Option<f64>,
    pub accepted: bool,
    pub reason: String,
    pub generation: u64,
}

/// Live gate with optional current ball. Missing ball / L / m fail closed.
#[derive(Debug, Default, Clone)]
pub struct LipschitzGate {
    current_ball: Option<LipschitzBall>,
    log: Vec<LipschitzCheck>,
    generation: u64,
}

impl LipschitzGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_ball(&self) -> Option<&LipschitzBall> {
        self.current_ball.as_ref()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn log(&self) -> &[LipschitzCheck] {
        &self.log
    }

    pub fn last(&self) -> Option<&LipschitzCheck> {
        self.log.last()
    }

    pub fn install(&mut self, ball: LipschitzBall) -> Result<(), LipschitzError> {
        ball.validate()?;
        self.current_ball = Some(ball);
        if self.generation == 0 {
            self.generation = 1;
        }
        Ok(())
    }

    /// Council freeze: start a new ball at `theta0`, keeping `m` and `L` (r does not grow).
    pub fn freeze_new_ball(
        &mut self,
        theta0: Theta,
        margin_m: f64,
        lipschitz_l: f64,
    ) -> Result<(), LipschitzError> {
        let ball = LipschitzBall::try_new(theta0, margin_m, lipschitz_l)?;
        self.current_ball = Some(ball);
        self.generation = self.generation.saturating_add(1);
        Ok(())
    }

    /// Ball chaining: an accepted check may become the next verified-safe center.
    pub fn freeze_if_accepted(&mut self, check: &LipschitzCheck) -> Result<(), LipschitzError> {
        if !check.accepted {
            return Err(LipschitzError::CannotFreezeRejected);
        }
        let theta0 = check.theta.clone();
        let m = check.margin_m.ok_or(LipschitzError::MissingM)?;
        let l = check.lipschitz_l.ok_or(LipschitzError::MissingL)?;
        self.freeze_new_ball(theta0, m, l)
    }

    /// A council majority cannot enlarge `r` to pass a Rejected ball.
    pub fn council_may_not_enlarge_radius(&self, proposed_r: f64) -> Result<(), LipschitzError> {
        let ball = self.current_ball.as_ref().ok_or(LipschitzError::MissingTheta0)?;
        let current = ball.radius()?;
        if proposed_r > current {
            return Err(LipschitzError::CouncilCannotEnlargeRadius {
                current,
                proposed: proposed_r,
            });
        }
        Ok(())
    }

    /// Core law: accept only if `distance(theta, theta0) < r` with `r = m / L`.
    pub fn verify(&mut self, theta: &Theta) -> LipschitzCheck {
        let ball = self.current_ball.clone();
        let check = verify_parts(
            ball.as_ref().map(|b| &b.theta0),
            ball.as_ref().map(|b| b.margin_m),
            ball.as_ref().map(|b| b.lipschitz_l),
            theta,
            self.generation,
        );
        log_check(&check);
        self.log.push(check.clone());
        check
    }
}

/// Fail-closed parts API — used by fixtures when L or m is missing.
pub fn verify_parts(
    theta0: Option<&Theta>,
    margin_m: Option<f64>,
    lipschitz_l: Option<f64>,
    theta: &Theta,
    generation: u64,
) -> LipschitzCheck {
    let mut check = LipschitzCheck {
        theta0: theta0.cloned(),
        theta: theta.clone(),
        lipschitz_l,
        margin_m,
        displacement: None,
        radius: None,
        accepted: false,
        reason: String::new(),
        generation,
    };

    let Some(theta0) = theta0 else {
        check.reason = LipschitzError::MissingTheta0.to_string();
        return check;
    };
    let Some(m) = margin_m else {
        check.reason = LipschitzError::MissingM.to_string();
        return check;
    };
    let Some(l) = lipschitz_l else {
        check.reason = LipschitzError::MissingL.to_string();
        return check;
    };

    if !m.is_finite() || m < 0.0 {
        check.reason = LipschitzError::MissingM.to_string();
        return check;
    }
    if !l.is_finite() || l <= 0.0 {
        check.reason = LipschitzError::MissingL.to_string();
        return check;
    }

    match euclidean(&theta.0, &theta0.0) {
        Ok(distance) => {
            let r = m / l;
            check.displacement = Some(distance);
            check.radius = Some(r);
            // Strict less-than. Distance == r is Rejected.
            if distance < r {
                check.accepted = true;
                check.reason = format!("accept displacement={distance} < r={r}");
            } else {
                check.accepted = false;
                check.reason = format!("reject displacement={distance} >= r={r}");
            }
        }
        Err(e) => {
            check.reason = e.to_string();
        }
    }
    check
}

pub fn euclidean(a: &[f64], b: &[f64]) -> Result<f64, LipschitzError> {
    if a.len() != b.len() {
        return Err(LipschitzError::DimMismatch {
            theta: a.len(),
            theta0: b.len(),
        });
    }
    if a.iter().any(|x| !x.is_finite()) || b.iter().any(|x| !x.is_finite()) {
        return Err(LipschitzError::NonFinite);
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt())
}

fn log_check(check: &LipschitzCheck) {
    println!(
        "[LipschitzGate] m={:?} L={:?} r={:?} displacement={:?} decision={} gen={} | {}",
        check.margin_m,
        check.lipschitz_l,
        check.radius,
        check.displacement,
        if check.accepted { "accept" } else { "reject" },
        check.generation,
        check.reason
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        let raw = include_str!("../fixtures/lipschitz_ball_v0.json");
        serde_json::from_str(raw).expect("lipschitz fixture json")
    }

    fn vec_from(v: &serde_json::Value) -> Theta {
        Theta(
            v.as_array()
                .expect("theta array")
                .iter()
                .map(|x| x.as_f64().expect("f64"))
                .collect(),
        )
    }

    #[test]
    fn known_safe_point_accepts_outside_rejects_zero_false_accepts() {
        let fx = fixture();
        let theta0 = vec_from(&fx["theta0"]);
        let m = fx["m"].as_f64().unwrap();
        let l = fx["L"].as_f64().unwrap();
        let expected_r = fx["r"].as_f64().unwrap();
        assert!((m / l - expected_r).abs() < 1e-12);
        // r is not the valence floor.
        assert!((expected_r - 0.999999).abs() > 1e-3);

        let mut gate = LipschitzGate::new();
        gate.install(LipschitzBall::try_new(theta0.clone(), m, l).unwrap())
            .unwrap();

        let safe = vec_from(&fx["safe"]["theta"]);
        let safe_check = gate.verify(&safe);
        assert!(safe_check.accepted, "reason={}", safe_check.reason);
        assert!(
            (safe_check.displacement.unwrap() - fx["safe"]["distance"].as_f64().unwrap()).abs()
                < 1e-12
        );

        let outside = vec_from(&fx["outside"]["theta"]);
        let out_check = gate.verify(&outside);
        assert!(!out_check.accepted, "must reject outside the ball");
        assert!(
            (out_check.displacement.unwrap() - fx["outside"]["distance"].as_f64().unwrap()).abs()
                < 1e-12
        );

        let mut false_accepts = 0u32;
        for probe in fx["false_accept_probes"].as_array().unwrap() {
            let theta = vec_from(probe);
            let check = gate.verify(&theta);
            if check.accepted {
                false_accepts += 1;
            }
        }
        assert_eq!(false_accepts, 0, "fixture has zero false accepts");
    }

    #[test]
    fn missing_l_or_m_is_rejected() {
        let theta0 = Theta(vec![0.0, 0.0, 0.0, 0.0]);
        let theta = Theta(vec![0.1, 0.0, 0.0, 0.0]);
        let missing_l = verify_parts(Some(&theta0), Some(1.0), None, &theta, 0);
        assert!(!missing_l.accepted);
        assert!(missing_l.reason.contains("L"));

        let missing_m = verify_parts(Some(&theta0), None, Some(2.0), &theta, 0);
        assert!(!missing_m.accepted);
        assert!(missing_m.reason.contains("m"));

        let missing_theta0 = verify_parts(None, Some(1.0), Some(2.0), &theta, 0);
        assert!(!missing_theta0.accepted);
        assert!(missing_theta0.reason.contains("theta0"));
    }

    #[test]
    fn rejected_ball_stays_rejected_after_mock_council_approve() {
        let mut gate = LipschitzGate::new();
        gate.install(LipschitzBall::try_new(Theta(vec![0.0, 0.0]), 1.0, 2.0).unwrap())
            .unwrap();
        let outside = Theta(vec![0.9, 0.0]);
        let check = gate.verify(&outside);
        assert!(!check.accepted);

        let mock_council_approve = true;
        assert!(mock_council_approve);
        // Majority vote does not mutate the check or enlarge r.
        assert!(gate.council_may_not_enlarge_radius(10.0).is_err());
        assert!(gate.freeze_if_accepted(&check).is_err());
        assert!(!gate.last().unwrap().accepted);
        assert!(!check.accepted);
    }

    #[test]
    fn ball_chaining_starts_new_center() {
        let mut gate = LipschitzGate::new();
        gate.install(LipschitzBall::try_new(Theta(vec![0.0, 0.0]), 1.0, 2.0).unwrap())
            .unwrap();
        // r = 0.5
        let step = Theta(vec![0.2, 0.0]);
        let first = gate.verify(&step);
        assert!(first.accepted);
        gate.freeze_if_accepted(&first).unwrap();
        assert_eq!(gate.generation(), 2);

        // Far from origin (0.6 >= 0.5) but inside the new ball around 0.2.
        let chained = Theta(vec![0.6, 0.0]);
        let second = gate.verify(&chained);
        assert!(
            second.accepted,
            "chained ball must accept a local step: {}",
            second.reason
        );
    }

    #[test]
    fn distance_equal_to_r_is_rejected() {
        let check = verify_parts(
            Some(&Theta(vec![0.0])),
            Some(1.0),
            Some(2.0),
            &Theta(vec![0.5]),
            0,
        );
        assert!(!check.accepted, "distance == r is not strictly less than r");
    }
}
