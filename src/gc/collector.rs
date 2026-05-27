use std::collections::HashMap;
use crate::vm::stack::Value;
use crate::vm::heap::Heap;

#[derive(Debug)]
pub struct GarbageCollector {
    heap: Heap,
    stats: GCStats,
    strategy: GCStrategy,
}

#[derive(Debug, Clone)]
pub struct GCStats {
    pub total_collections: usize,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_live: usize,
    pub peak_memory: usize,
    pub collection_time_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum GCStrategy {
    MarkAndSweep,
    Generational,
    ReferenceCounting,
    Hybrid,
}

impl GarbageCollector {
    pub fn new() -> Self {
        Self {
            heap: Heap::new(),
            stats: GCStats {
                total_collections: 0,
                total_allocated: 0,
                total_freed: 0,
                current_live: 0,
                peak_memory: 0,
                collection_time_ms: 0.0,
            },
            strategy: GCStrategy::MarkAndSweep,
        }
    }

    pub fn with_strategy(mut self, strategy: GCStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn allocate(&mut self, value: Value) -> usize {
        let index = self.heap.allocate(value);
        self.stats.total_allocated += 1;
        self.stats.current_live += 1;
        if self.stats.current_live > self.stats.peak_memory {
            self.stats.peak_memory = self.stats.current_live;
        }
        index
    }

    pub fn collect(&mut self, roots: &[Value]) {
        let start = std::time::Instant::now();

        match self.strategy {
            GCStrategy::MarkAndSweep => self.mark_and_sweep(roots),
            GCStrategy::Generational => self.generational_collect(roots),
            GCStrategy::ReferenceCounting => self.reference_counting_collect(),
            GCStrategy::Hybrid => self.hybrid_collect(roots),
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.collection_time_ms += elapsed;
        self.stats.total_collections += 1;
    }

    fn mark_and_sweep(&mut self, roots: &[Value]) {
        // Mark phase
        for root in roots {
            self.mark_value(root);
        }

        // Sweep phase
        let before = self.heap.allocated_count();
        self.heap.sweep();
        let after = self.heap.allocated_count();
        let freed = before.saturating_sub(after);

        self.stats.total_freed += freed;
        self.stats.current_live = after;
    }

    fn generational_collect(&mut self, roots: &[Value]) {
        // Simplified generational GC
        // Young generation: recently allocated objects
        // Old generation: objects that survived multiple collections

        // For now, just do mark and sweep
        self.mark_and_sweep(roots);
    }

    fn reference_counting_collect(&mut self) {
        // Reference counting doesn't need roots
        // Objects are freed when their reference count drops to 0
        // This is handled by the IncRef/DecRef instructions
    }

    fn hybrid_collect(&mut self, roots: &[Value]) {
        // Use reference counting for most objects
        // Use mark and sweep periodically to handle cycles
        if self.stats.total_collections % 10 == 0 {
            self.mark_and_sweep(roots);
        }
    }

    fn mark_value(&mut self, value: &Value) {
        match value {
            Value::Array(elements) => {
                for elem in elements {
                    self.mark_value(elem);
                }
            }
            Value::Map(entries) => {
                for (k, v) in entries {
                    self.mark_value(k);
                    self.mark_value(v);
                }
            }
            Value::Tuple(elements) => {
                for elem in elements {
                    self.mark_value(elem);
                }
            }
            Value::Object(obj) => {
                for (_, v) in &obj.fields {
                    self.mark_value(v);
                }
            }
            Value::Function(func) => {
                for upvalue in &func.upvalues {
                    self.mark_value(upvalue);
                }
            }
            _ => {}
        }
    }

    pub fn should_collect(&self) -> bool {
        self.heap.should_gc()
    }

    pub fn get_stats(&self) -> &GCStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = GCStats {
            total_collections: 0,
            total_allocated: 0,
            total_freed: 0,
            current_live: 0,
            peak_memory: 0,
            collection_time_ms: 0.0,
        };
    }

    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    pub fn heap_mut(&mut self) -> &mut Heap {
        &mut self.heap
    }
}
