use std::time::{Duration, Instant};
use tokio::time::sleep;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    state: CircuitState,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout_secs: u64) -> Self {
        Self {
            failure_threshold,
            recovery_timeout: Duration::from_secs(recovery_timeout_secs),
            failure_count: 0,
            last_failure_time: None,
            state: CircuitState::Closed,
        }
    }

    pub async fn call<F, Fut, T, E>(&mut self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        if self.state == CircuitState::Open {
            if let Some(last_failure) = self.last_failure_time {
                if last_failure.elapsed() > self.recovery_timeout {
                    self.state = CircuitState::HalfOpen;
                } else {
                    return Err(From::from("Circuit breaker is OPEN"));
                }
            }
        }

        match f().await {
            Ok(result) => {
                if self.state == CircuitState::HalfOpen {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                }
                Ok(result)
            }
            Err(e) => {
                self.failure_count += 1;
                self.last_failure_time = Some(Instant::now());

                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
                Err(e)
            }
        }
    }
}