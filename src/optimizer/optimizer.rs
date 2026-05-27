use crate::compiler::bytecode::{Bytecode, Instruction, Constant};
use crate::errors::OmegaResult;
use super::passes::*;

pub struct Optimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
    debug: bool,
}

impl Optimizer {
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(ConstantFolding),
                Box::new(DeadCodeElimination),
                Box::new(CommonSubexpressionElimination),
                Box::new(PeepholeOptimizer),
                Box::new(StrengthReduction),
                Box::new(LoopInvariantCodeMotion),
                Box::new(CopyPropagation),
                Box::new(InlineSmallFunctions),
            ],
            max_iterations: 10,
            debug: false,
        }
    }

    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn optimize(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut changed = true;
        let mut iteration = 0;

        while changed && iteration < self.max_iterations {
            changed = false;
            iteration += 1;

            for pass in &self.passes {
                let pass_stats = pass.run(bytecode)?;
                if pass_stats.instructions_removed > 0
                    || pass_stats.instructions_modified > 0
                    || pass_stats.constants_removed > 0
                {
                    changed = true;
                    stats.merge(&pass_stats);

                    if self.debug {
                        eprintln!(
                            "  Pass '{}' in iteration {}: {} removed, {} modified",
                            pass.name(),
                            iteration,
                            pass_stats.instructions_removed,
                            pass_stats.instructions_modified
                        );
                    }
                }
            }
        }

        stats.iterations = iteration;
        Ok(stats)
    }

    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    pub fn remove_pass(&mut self, name: &str) {
        self.passes.retain(|p| p.name() != name);
    }
}

#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    pub iterations: usize,
    pub instructions_removed: usize,
    pub instructions_modified: usize,
    pub constants_removed: usize,
    pub functions_inlined: usize,
    pub time_ms: f64,
}

impl OptimizationStats {
    pub fn merge(&mut self, other: &OptimizationStats) {
        self.instructions_removed += other.instructions_removed;
        self.instructions_modified += other.instructions_modified;
        self.constants_removed += other.constants_removed;
        self.functions_inlined += other.functions_inlined;
    }

    pub fn total_changes(&self) -> usize {
        self.instructions_removed + self.instructions_modified + self.constants_removed
    }
}

pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats>;
}
