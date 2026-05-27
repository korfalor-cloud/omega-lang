use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub self_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub avg_time: Duration,
    pub children: HashMap<String, ProfileEntry>,
}

impl ProfileEntry {
    pub fn new(name: String) -> Self {
        Self {
            name,
            call_count: 0,
            total_time: Duration::ZERO,
            self_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            children: HashMap::new(),
        }
    }

    pub fn record(&mut self, duration: Duration) {
        self.call_count += 1;
        self.total_time += duration;
        if duration < self.min_time {
            self.min_time = duration;
        }
        if duration > self.max_time {
            self.max_time = duration;
        }
        self.avg_time = self.total_time / self.call_count as u32;
    }
}

#[derive(Debug)]
pub struct Profiler {
    entries: HashMap<String, ProfileEntry>,
    call_stack: Vec<ProfileFrame>,
    enabled: bool,
    start_time: Instant,
    total_instructions: usize,
    instruction_counts: HashMap<String, usize>,
    memory_snapshots: Vec<MemorySnapshot>,
}

#[derive(Debug)]
struct ProfileFrame {
    name: String,
    start_time: Instant,
    children_time: Duration,
}

#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Duration,
    pub heap_size: usize,
    pub stack_size: usize,
    pub live_objects: usize,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            call_stack: Vec::new(),
            enabled: true,
            start_time: Instant::now(),
            total_instructions: 0,
            instruction_counts: HashMap::new(),
            memory_snapshots: Vec::new(),
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn enter_function(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        self.call_stack.push(ProfileFrame {
            name: name.to_string(),
            start_time: Instant::now(),
            children_time: Duration::ZERO,
        });
    }

    pub fn exit_function(&mut self) {
        if !self.enabled {
            return;
        }

        if let Some(frame) = self.call_stack.pop() {
            let elapsed = frame.start_time.elapsed();

            let entry = self
                .entries
                .entry(frame.name.clone())
                .or_insert_with(|| ProfileEntry::new(frame.name.clone()));
            entry.record(elapsed);

            // Add self time (total - children)
            let self_time = elapsed - frame.children_time;
            entry.self_time += self_time;

            // Notify parent about children time
            if let Some(parent) = self.call_stack.last_mut() {
                parent.children_time += elapsed;
            }
        }
    }

    pub fn record_instruction(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        self.total_instructions += 1;
        *self.instruction_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    pub fn take_memory_snapshot(&mut self, heap_size: usize, stack_size: usize, live_objects: usize) {
        if !self.enabled {
            return;
        }

        self.memory_snapshots.push(MemorySnapshot {
            timestamp: self.start_time.elapsed(),
            heap_size,
            stack_size,
            live_objects,
        });
    }

    pub fn get_entry(&self, name: &str) -> Option<&ProfileEntry> {
        self.entries.get(name)
    }

    pub fn entries(&self) -> &HashMap<String, ProfileEntry> {
        &self.entries
    }

    pub fn total_instructions(&self) -> usize {
        self.total_instructions
    }

    pub fn instruction_counts(&self) -> &HashMap<String, usize> {
        &self.instruction_counts
    }

    pub fn memory_snapshots(&self) -> &[MemorySnapshot] {
        &self.memory_snapshots
    }

    pub fn reset(&mut self) {
        self.entries.clear();
        self.call_stack.clear();
        self.total_instructions = 0;
        self.instruction_counts.clear();
        self.memory_snapshots.clear();
        self.start_time = Instant::now();
    }

    pub fn report(&self) -> ProfileReport {
        let mut sorted_entries: Vec<&ProfileEntry> = self.entries.values().collect();
        sorted_entries.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        let top_functions: Vec<FunctionProfile> = sorted_entries
            .iter()
            .take(20)
            .map(|e| FunctionProfile {
                name: e.name.clone(),
                call_count: e.call_count,
                total_time_ms: e.total_time.as_secs_f64() * 1000.0,
                self_time_ms: e.self_time.as_secs_f64() * 1000.0,
                avg_time_ms: e.avg_time.as_secs_f64() * 1000.0,
                min_time_ms: e.min_time.as_secs_f64() * 1000.0,
                max_time_ms: e.max_time.as_secs_f64() * 1000.0,
            })
            .collect();

        let mut sorted_instructions: Vec<(&String, &usize)> = self.instruction_counts.iter().collect();
        sorted_instructions.sort_by(|a, b| b.1.cmp(a.1));

        let top_instructions: Vec<InstructionProfile> = sorted_instructions
            .iter()
            .take(10)
            .map(|(name, count)| InstructionProfile {
                name: (*name).clone(),
                count: **count,
                percentage: (**count as f64 / self.total_instructions as f64) * 100.0,
            })
            .collect();

        let total_runtime_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;

        ProfileReport {
            total_runtime_ms,
            total_instructions: self.total_instructions,
            instructions_per_second: if total_runtime_ms > 0.0 {
                (self.total_instructions as f64 / total_runtime_ms * 1000.0) as u64
            } else {
                0
            },
            top_functions,
            top_instructions,
            memory_snapshots: self.memory_snapshots.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ProfileReport {
    pub total_runtime_ms: f64,
    pub total_instructions: usize,
    pub instructions_per_second: u64,
    pub top_functions: Vec<FunctionProfile>,
    pub top_instructions: Vec<InstructionProfile>,
    pub memory_snapshots: Vec<MemorySnapshot>,
}

#[derive(Debug)]
pub struct FunctionProfile {
    pub name: String,
    pub call_count: usize,
    pub total_time_ms: f64,
    pub self_time_ms: f64,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
}

#[derive(Debug)]
pub struct InstructionProfile {
    pub name: String,
    pub count: usize,
    pub percentage: f64,
}

impl std::fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Profile Report ===")?;
        writeln!(f, "Total runtime: {:.2} ms", self.total_runtime_ms)?;
        writeln!(f, "Total instructions: {}", self.total_instructions)?;
        writeln!(f, "Instructions/second: {}", self.instructions_per_second)?;
        writeln!(f)?;

        if !self.top_functions.is_empty() {
            writeln!(f, "=== Top Functions ===")?;
            writeln!(
                f,
                "{:<30} {:>8} {:>12} {:>12} {:>12}",
                "Name", "Calls", "Total (ms)", "Self (ms)", "Avg (ms)"
            )?;
            writeln!(f, "{}", "-".repeat(80))?;
            for func in &self.top_functions {
                writeln!(
                    f,
                    "{:<30} {:>8} {:>12.2} {:>12.2} {:>12.4}",
                    func.name, func.call_count, func.total_time_ms, func.self_time_ms, func.avg_time_ms
                )?;
            }
            writeln!(f)?;
        }

        if !self.top_instructions.is_empty() {
            writeln!(f, "=== Instruction Distribution ===")?;
            writeln!(f, "{:<20} {:>12} {:>10}", "Instruction", "Count", "%" )?;
            writeln!(f, "{}", "-".repeat(45))?;
            for inst in &self.top_instructions {
                writeln!(
                    f,
                    "{:<20} {:>12} {:>9.1}%",
                    inst.name, inst.count, inst.percentage
                )?;
            }
        }

        Ok(())
    }
}

// Hot path detection
pub struct HotPathDetector {
    paths: HashMap<Vec<String>, usize>,
    current_path: Vec<String>,
    min_frequency: usize,
}

impl HotPathDetector {
    pub fn new(min_frequency: usize) -> Self {
        Self {
            paths: HashMap::new(),
            current_path: Vec::new(),
            min_frequency,
        }
    }

    pub fn push(&mut self, name: &str) {
        self.current_path.push(name.to_string());
        let path = self.current_path.clone();
        *self.paths.entry(path).or_insert(0) += 1;
    }

    pub fn pop(&mut self) {
        self.current_path.pop();
    }

    pub fn hot_paths(&self) -> Vec<(&Vec<String>, &usize)> {
        self.paths
            .iter()
            .filter(|(_, &count)| count >= self.min_frequency)
            .collect()
    }

    pub fn reset(&mut self) {
        self.paths.clear();
        self.current_path.clear();
    }
}

// Memory profiler
pub struct MemoryProfiler {
    allocations: HashMap<String, AllocationInfo>,
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    current_usage: usize,
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub count: usize,
    pub total_bytes: usize,
    pub avg_bytes: usize,
    pub peak_bytes: usize,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            total_freed: 0,
            peak_usage: 0,
            current_usage: 0,
        }
    }

    pub fn record_allocation(&mut self, type_name: &str, bytes: usize) {
        self.total_allocated += bytes;
        self.current_usage += bytes;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }

        let entry = self
            .allocations
            .entry(type_name.to_string())
            .or_insert_with(|| AllocationInfo {
                count: 0,
                total_bytes: 0,
                avg_bytes: 0,
                peak_bytes: 0,
            });

        entry.count += 1;
        entry.total_bytes += bytes;
        entry.avg_bytes = entry.total_bytes / entry.count;
        if bytes > entry.peak_bytes {
            entry.peak_bytes = bytes;
        }
    }

    pub fn record_deallocation(&mut self, type_name: &str, bytes: usize) {
        self.total_freed += bytes;
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    pub fn total_freed(&self) -> usize {
        self.total_freed
    }

    pub fn allocations(&self) -> &HashMap<String, AllocationInfo> {
        &self.allocations
    }

    pub fn report(&self) -> String {
        let mut output = String::from("=== Memory Profile ===\n");
        output.push_str(&format!("Current usage: {} bytes\n", self.current_usage));
        output.push_str(&format!("Peak usage: {} bytes\n", self.peak_usage));
        output.push_str(&format!("Total allocated: {} bytes\n", self.total_allocated));
        output.push_str(&format!("Total freed: {} bytes\n", self.total_freed));
        output.push_str(&format!(
            "Leaked: {} bytes\n\n",
            self.total_allocated - self.total_freed
        ));

        let mut sorted: Vec<(&String, &AllocationInfo)> = self.allocations.iter().collect();
        sorted.sort_by(|a, b| b.1.total_bytes.cmp(&a.1.total_bytes));

        output.push_str(&format!(
            "{:<20} {:>8} {:>12} {:>12} {:>12}\n",
            "Type", "Count", "Total", "Avg", "Peak"
        ));
        output.push_str(&format!("{}\n", "-".repeat(70)));

        for (name, info) in sorted.iter().take(20) {
            output.push_str(&format!(
                "{:<20} {:>8} {:>12} {:>12} {:>12}\n",
                name, info.count, info.total_bytes, info.avg_bytes, info.peak_bytes
            ));
        }

        output
    }
}
