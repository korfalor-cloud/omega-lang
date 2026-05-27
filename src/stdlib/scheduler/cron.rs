/// Cron expression parser and task scheduler.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CronExpression {
    pub minute: CronField,
    pub hour: CronField,
    pub day_of_month: CronField,
    pub month: CronField,
    pub day_of_week: CronField,
}

#[derive(Debug, Clone)]
pub enum CronField {
    Any,
    Exact(u32),
    List(Vec<u32>),
    Range(u32, u32),
    Step(u32),
    EveryN(u32),
}

impl CronField {
    pub fn matches(&self, value: u32) -> bool {
        match self {
            CronField::Any => true,
            CronField::Exact(v) => *v == value,
            CronField::List(values) => values.contains(&value),
            CronField::Range(start, end) => value >= *start && value <= *end,
            CronField::Step(step) => value % step == 0,
            CronField::EveryN(n) => value % n == 0,
        }
    }
}

impl CronExpression {
    pub fn parse(expression: &str) -> Result<Self, String> {
        let parts: Vec<&str> = expression.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(format!("Expected 5 fields, got {}", parts.len()));
        }

        Ok(Self {
            minute: parse_field(parts[0], 0, 59)?,
            hour: parse_field(parts[1], 0, 23)?,
            day_of_month: parse_field(parts[2], 1, 31)?,
            month: parse_field(parts[3], 1, 12)?,
            day_of_week: parse_field(parts[4], 0, 7)?,
        })
    }

    pub fn matches(&self, minute: u32, hour: u32, day: u32, month: u32, weekday: u32) -> bool {
        self.minute.matches(minute)
            && self.hour.matches(hour)
            && self.day_of_month.matches(day)
            && self.month.matches(month)
            && self.day_of_week.matches(weekday % 7)
    }

    /// Get the next N matching timestamps from a given time.
    pub fn next_n(&self, from_minute: u32, from_hour: u32, from_day: u32, from_month: u32, n: usize) -> Vec<(u32, u32, u32, u32)> {
        let mut results = Vec::new();
        let mut minute = from_minute;
        let mut hour = from_hour;
        let mut day = from_day;
        let mut month = from_month;
        let mut skip_first = true;

        for _ in 0..n * 1000 { // Safety limit
            if self.minute.matches(minute)
                && self.hour.matches(hour)
                && self.day_of_month.matches(day)
                && self.month.matches(month)
            {
                if skip_first {
                    skip_first = false;
                } else {
                    results.push((minute, hour, day, month));
                    if results.len() >= n {
                        break;
                    }
                }
            }

            minute += 1;
            if minute >= 60 {
                minute = 0;
                hour += 1;
                if hour >= 24 {
                    hour = 0;
                    day += 1;
                    if day > 31 {
                        day = 1;
                        month += 1;
                        if month > 12 {
                            month = 1;
                        }
                    }
                }
            }
        }

        results
    }

    pub fn to_string(&self) -> String {
        format!("{} {} {} {} {}",
            field_to_string(&self.minute),
            field_to_string(&self.hour),
            field_to_string(&self.day_of_month),
            field_to_string(&self.month),
            field_to_string(&self.day_of_week),
        )
    }
}

fn field_to_string(field: &CronField) -> String {
    match field {
        CronField::Any => "*".to_string(),
        CronField::Exact(v) => v.to_string(),
        CronField::List(v) => v.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        CronField::Range(s, e) => format!("{}-{}", s, e),
        CronField::Step(s) => format!("*/{}", s),
        CronField::EveryN(n) => format!("*/{}", n),
    }
}

fn parse_field(s: &str, min: u32, max: u32) -> Result<CronField, String> {
    if s == "*" {
        return Ok(CronField::Any);
    }

    if s.contains(',') {
        let values: Result<Vec<u32>, _> = s.split(',').map(|v| v.parse::<u32>()).collect();
        let values = values.map_err(|e| format!("Invalid list: {}", e))?;
        for &v in &values {
            if v < min || v > max {
                return Err(format!("Value {} out of range [{}, {}]", v, min, max));
            }
        }
        return Ok(CronField::List(values));
    }

    if s.contains('-') {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid range: {}", s));
        }
        let start: u32 = parts[0].parse().map_err(|_| format!("Invalid range start: {}", parts[0]))?;
        let end: u32 = parts[1].parse().map_err(|_| format!("Invalid range end: {}", parts[1]))?;
        if start < min || end > max || start > end {
            return Err(format!("Invalid range: {}-{}", start, end));
        }
        return Ok(CronField::Range(start, end));
    }

    if s.starts_with("*/") {
        let step: u32 = s[2..].parse().map_err(|_| format!("Invalid step: {}", s))?;
        if step == 0 || step > max {
            return Err(format!("Step {} out of range", step));
        }
        return Ok(CronField::Step(step));
    }

    let value: u32 = s.parse().map_err(|_| format!("Invalid value: {}", s))?;
    if value < min || value > max {
        return Err(format!("Value {} out of range [{}, {}]", value, min, max));
    }
    Ok(CronField::Exact(value))
}

/// Task scheduler that runs tasks at specified intervals.
#[derive(Debug)]
pub struct Scheduler {
    tasks: Vec<ScheduledTask>,
    tick_count: u64,
}

#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: String,
    pub name: String,
    pub cron: CronExpression,
    pub enabled: bool,
    pub last_run: Option<u64>,
    pub run_count: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            tick_count: 0,
        }
    }

    pub fn add_task(&mut self, id: &str, name: &str, cron_expr: &str) -> Result<(), String> {
        let cron = CronExpression::parse(cron_expr)?;
        self.tasks.push(ScheduledTask {
            id: id.to_string(),
            name: name.to_string(),
            cron,
            enabled: true,
            last_run: None,
            run_count: 0,
        });
        Ok(())
    }

    pub fn remove_task(&mut self, id: &str) {
        self.tasks.retain(|t| t.id != id);
    }

    pub fn enable_task(&mut self, id: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.enabled = true;
        }
    }

    pub fn disable_task(&mut self, id: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.enabled = false;
        }
    }

    /// Check which tasks should run at the given time.
    pub fn tick(&mut self, minute: u32, hour: u32, day: u32, month: u32, weekday: u32) -> Vec<String> {
        self.tick_count += 1;
        let mut due = Vec::new();

        for task in &mut self.tasks {
            if !task.enabled {
                continue;
            }
            if task.cron.matches(minute, hour, day, month, weekday) {
                task.run_count += 1;
                task.last_run = Some(self.tick_count);
                due.push(task.id.clone());
            }
        }

        due
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn get_task(&self, id: &str) -> Option<&ScheduledTask> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn tasks(&self) -> &[ScheduledTask] {
        &self.tasks
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple interval-based timer (not cron).
#[derive(Debug, Clone)]
pub struct IntervalTimer {
    interval_ticks: u64,
    last_trigger: u64,
    tick_count: u64,
}

impl IntervalTimer {
    pub fn new(interval: u64) -> Self {
        Self {
            interval_ticks: interval,
            last_trigger: 0,
            tick_count: 0,
        }
    }

    pub fn tick(&mut self) -> bool {
        self.tick_count += 1;
        if self.tick_count - self.last_trigger >= self.interval_ticks {
            self.last_trigger = self.tick_count;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.tick_count = 0;
        self.last_trigger = 0;
    }

    pub fn ticks_until_next(&self) -> u64 {
        self.interval_ticks - (self.tick_count - self.last_trigger)
    }
}

/// Rate limiter using token bucket algorithm.
#[derive(Debug)]
pub struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: u64,
    tick: u64,
}

impl TokenBucket {
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: 0,
            tick: 0,
        }
    }

    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        self.tick += 1;
        let elapsed = self.tick - self.last_refill;
        self.tokens = (self.tokens + elapsed as f64 * self.refill_rate).min(self.max_tokens);
        self.last_refill = self.tick;
    }

    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }

    pub fn wait_time(&self, tokens: f64) -> u64 {
        if self.tokens >= tokens {
            0
        } else {
            ((tokens - self.tokens) / self.refill_rate).ceil() as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cron_parse() {
        let cron = CronExpression::parse("*/5 * * * *").unwrap();
        assert!(cron.minute.matches(0));
        assert!(cron.minute.matches(5));
        assert!(cron.minute.matches(10));
        assert!(!cron.minute.matches(3));
        assert!(cron.hour.matches(0));
        assert!(cron.hour.matches(23));
    }

    #[test]
    fn test_cron_specific() {
        let cron = CronExpression::parse("30 9 * * 1-5").unwrap();
        assert!(cron.matches(30, 9, 15, 6, 1));  // Monday 9:30
        assert!(!cron.matches(30, 9, 15, 6, 0)); // Sunday 9:30
    }

    #[test]
    fn test_cron_list() {
        let cron = CronExpression::parse("0 0 1,15 * *").unwrap();
        assert!(cron.day_of_month.matches(1));
        assert!(cron.day_of_month.matches(15));
        assert!(!cron.day_of_month.matches(10));
    }

    #[test]
    fn test_scheduler() {
        let mut scheduler = Scheduler::new();
        scheduler.add_task("backup", "Backup DB", "0 2 * * *").unwrap();
        scheduler.add_task("cleanup", "Cleanup logs", "*/30 * * * *").unwrap();

        assert_eq!(scheduler.task_count(), 2);

        let due = scheduler.tick(0, 2, 1, 1, 1);
        assert!(due.contains(&"backup".to_string()));
    }

    #[test]
    fn test_interval_timer() {
        let mut timer = IntervalTimer::new(5);
        for _ in 0..4 {
            assert!(!timer.tick());
        }
        assert!(timer.tick());
        assert!(!timer.tick());
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!(bucket.try_consume(5.0));
        assert!(bucket.try_consume(5.0));
        assert!(!bucket.try_consume(1.0));
    }
}
