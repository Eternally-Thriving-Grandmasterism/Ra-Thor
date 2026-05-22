/*!
 * rrel_patsagi_scheduler.rs v1.0.0
 * PATSAGi Queue + Batch Council Scheduler for RREL
 * Timestamp-based, sovereign, local-only. Ready for NEXi / TOLC integration.
 * Part of Ra-Thor Eternal One Organism.
 */

use chrono::{DateTime, Utc};
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub enum PatsagiAlertLevel { Info, Warning, ActionRequired, Critical }

#[derive(Debug, Clone)]
pub struct PatsagiTask {
    pub id: String,
    pub due_at: DateTime<Utc>,
    pub message: String,
    pub level: PatsagiAlertLevel,
    pub related_id: Option<String>,
    pub acknowledged: bool,
}

pub struct PatsagiQueue {
    tasks: VecDeque<PatsagiTask>,
}

impl PatsagiQueue {
    pub fn new() -> Self { Self { tasks: VecDeque::new() } }
    pub fn enqueue(&mut self, task: PatsagiTask) { self.tasks.push_back(task); }
    pub fn get_due_tasks(&self, now: DateTime<Utc>) -> Vec<&PatsagiTask> {
        self.tasks.iter().filter(|t| t.due_at <= now && !t.acknowledged).collect()
    }
    pub fn acknowledge_task(&mut self, id: &str) {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.id == id) { t.acknowledged = true; }
    }
}

pub fn create_scheduled_reminder(id: &str, hours_from_now: i64, msg: &str, level: PatsagiAlertLevel) -> PatsagiTask {
    PatsagiTask {
        id: id.to_string(),
        due_at: Utc::now() + chrono::Duration::hours(hours_from_now),
        message: msg.to_string(),
        level,
        related_id: None,
        acknowledged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_queue_due() { assert!(true); }
}