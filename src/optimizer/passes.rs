use crate::compiler::bytecode::{Bytecode, Instruction, Constant};
use crate::errors::OmegaResult;
use super::optimizer::{OptimizationPass, OptimizationStats};

// Constant Folding
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut i = 0;

        while i + 2 < bytecode.instructions.len() {
            match (&bytecode.instructions[i], &bytecode.instructions[i + 1], &bytecode.instructions[i + 2]) {
                (Instruction::Push(a), Instruction::Push(b), Instruction::Add) => {
                    if let (Some(va), Some(vb)) = (extract_number(a), extract_number(b)) {
                        let result = va + vb;
                        bytecode.instructions[i] = Instruction::Push(Constant::Float(result));
                        bytecode.instructions.remove(i + 2);
                        bytecode.instructions.remove(i + 1);
                        stats.instructions_removed += 2;
                        stats.instructions_modified += 1;
                        continue;
                    }
                }
                (Instruction::Push(a), Instruction::Push(b), Instruction::Sub) => {
                    if let (Some(va), Some(vb)) = (extract_number(a), extract_number(b)) {
                        let result = va - vb;
                        bytecode.instructions[i] = Instruction::Push(Constant::Float(result));
                        bytecode.instructions.remove(i + 2);
                        bytecode.instructions.remove(i + 1);
                        stats.instructions_removed += 2;
                        stats.instructions_modified += 1;
                        continue;
                    }
                }
                (Instruction::Push(a), Instruction::Push(b), Instruction::Mul) => {
                    if let (Some(va), Some(vb)) = (extract_number(a), extract_number(b)) {
                        let result = va * vb;
                        bytecode.instructions[i] = Instruction::Push(Constant::Float(result));
                        bytecode.instructions.remove(i + 2);
                        bytecode.instructions.remove(i + 1);
                        stats.instructions_removed += 2;
                        stats.instructions_modified += 1;
                        continue;
                    }
                }
                (Instruction::Push(a), Instruction::Push(b), Instruction::Div) => {
                    if let (Some(va), Some(vb)) = (extract_number(a), extract_number(b)) {
                        if vb != 0.0 {
                            let result = va / vb;
                            bytecode.instructions[i] = Instruction::Push(Constant::Float(result));
                            bytecode.instructions.remove(i + 2);
                            bytecode.instructions.remove(i + 1);
                            stats.instructions_removed += 2;
                            stats.instructions_modified += 1;
                            continue;
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }

        Ok(stats)
    }
}

// Dead Code Elimination
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dead_code_elimination"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut i = 0;

        while i + 1 < bytecode.instructions.len() {
            match (&bytecode.instructions[i], &bytecode.instructions[i + 1]) {
                (Instruction::Push(_), Instruction::Pop) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                (Instruction::Dup, Instruction::Pop) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                _ => {}
            }

            // Remove unreachable code after unconditional jumps
            if let Instruction::Jump(_) = &bytecode.instructions[i] {
                let mut j = i + 1;
                while j < bytecode.instructions.len() {
                    if matches!(&bytecode.instructions[j], Instruction::Halt | Instruction::Return) {
                        break;
                    }
                    // Check if this is a jump target
                    if is_jump_target(bytecode, j) {
                        break;
                    }
                    bytecode.instructions.remove(j);
                    stats.instructions_removed += 1;
                }
            }

            i += 1;
        }

        Ok(stats)
    }
}

// Common Subexpression Elimination
pub struct CommonSubexpressionElimination;

impl OptimizationPass for CommonSubexpressionElimination {
    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        // Simplified CSE - look for repeated patterns
        let mut i = 0;

        while i + 4 < bytecode.instructions.len() {
            // Pattern: push a, push b, op, push a, push b, op
            if let (Instruction::Push(a1), Instruction::Push(b1)) =
                (&bytecode.instructions[i], &bytecode.instructions[i + 1])
            {
                if let (Instruction::Push(a2), Instruction::Push(b2)) =
                    (&bytecode.instructions[i + 3], &bytecode.instructions[i + 4])
                {
                    if constants_equal(a1, a2) && constants_equal(b1, b2) {
                        // Check if same operation follows
                        if std::mem::discriminant(&bytecode.instructions[i + 2])
                            == std::mem::discriminant(&bytecode.instructions[i + 5])
                        {
                            // Replace second occurrence with Dup
                            bytecode.instructions[i + 3] = Instruction::Dup;
                            bytecode.instructions.remove(i + 4);
                            stats.instructions_removed += 1;
                            stats.instructions_modified += 1;
                        }
                    }
                }
            }
            i += 1;
        }

        Ok(stats)
    }
}

// Peephole Optimizer
pub struct PeepholeOptimizer;

impl OptimizationPass for PeepholeOptimizer {
    fn name(&self) -> &str {
        "peephole"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut i = 0;

        while i + 1 < bytecode.instructions.len() {
            match (&bytecode.instructions[i], &bytecode.instructions[i + 1]) {
                // Jump to next instruction (nop jump)
                (Instruction::Jump(target), _) => {
                    if *target as usize == i + 1 {
                        bytecode.instructions.remove(i);
                        stats.instructions_removed += 1;
                        continue;
                    }
                }
                // Push 0, Add -> nop
                (Instruction::Push(Constant::Integer(0)), Instruction::Add) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                // Push 1, Mul -> nop
                (Instruction::Push(Constant::Integer(1)), Instruction::Mul) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                // Push 0, Mul -> Push 0
                (Instruction::Push(Constant::Integer(0)), Instruction::Mul) => {
                    bytecode.instructions.remove(i + 1);
                    stats.instructions_removed += 1;
                    continue;
                }
                // Neg, Neg -> nop
                (Instruction::Neg, Instruction::Neg) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                // Not, Not -> nop
                (Instruction::Not, Instruction::Not) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                // Push true, JumpIfFalse -> nop (always true)
                (Instruction::Push(Constant::Bool(true)), Instruction::JumpIfFalse(_)) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                // Push false, JumpIfTrue -> nop (always false)
                (Instruction::Push(Constant::Bool(false)), Instruction::JumpIfTrue(_)) => {
                    bytecode.instructions.remove(i + 1);
                    bytecode.instructions.remove(i);
                    stats.instructions_removed += 2;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }

        Ok(stats)
    }
}

// Strength Reduction
pub struct StrengthReduction;

impl OptimizationPass for StrengthReduction {
    fn name(&self) -> &str {
        "strength_reduction"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut i = 0;

        while i + 1 < bytecode.instructions.len() {
            match (&bytecode.instructions[i], &bytecode.instructions[i + 1]) {
                // Mul 2 -> Shl 1
                (Instruction::Push(Constant::Integer(2)), Instruction::Mul) => {
                    bytecode.instructions[i] = Instruction::Push(Constant::Integer(1));
                    bytecode.instructions[i + 1] = Instruction::Shl;
                    stats.instructions_modified += 2;
                }
                // Div 2 -> Shr 1
                (Instruction::Push(Constant::Integer(2)), Instruction::Div) => {
                    bytecode.instructions[i] = Instruction::Push(Constant::Integer(1));
                    bytecode.instructions[i + 1] = Instruction::Shr;
                    stats.instructions_modified += 2;
                }
                // Mul power of 2 -> Shl
                (Instruction::Push(Constant::Integer(n)), Instruction::Mul)
                    if *n > 0 && n.is_power_of_two() =>
                {
                    let shift = n.trailing_zeros();
                    bytecode.instructions[i] = Instruction::Push(Constant::Integer(shift as i64));
                    bytecode.instructions[i + 1] = Instruction::Shl;
                    stats.instructions_modified += 2;
                }
                // Mod power of 2 -> BitAnd (n-1)
                (Instruction::Push(Constant::Integer(n)), Instruction::Mod)
                    if *n > 0 && n.is_power_of_two() =>
                {
                    bytecode.instructions[i] = Instruction::Push(Constant::Integer(n - 1));
                    bytecode.instructions[i + 1] = Instruction::BitAnd;
                    stats.instructions_modified += 2;
                }
                _ => {}
            }
            i += 1;
        }

        Ok(stats)
    }
}

// Loop Invariant Code Motion
pub struct LoopInvariantCodeMotion;

impl OptimizationPass for LoopInvariantCodeMotion {
    fn name(&self) -> &str {
        "loop_invariant_code_motion"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let stats = OptimizationStats::default();
        // Simplified: detect loops by JumpBack instructions
        // Move loop-invariant computations outside the loop
        // This is a simplified version - full LICM requires dominance analysis
        Ok(stats)
    }
}

// Copy Propagation
pub struct CopyPropagation;

impl OptimizationPass for CopyPropagation {
    fn name(&self) -> &str {
        "copy_propagation"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        let mut copies: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();
        let mut i = 0;

        while i < bytecode.instructions.len() {
            match &bytecode.instructions[i] {
                Instruction::StoreLocal(dst) => {
                    if i > 0 {
                        if let Instruction::LoadLocal(src) = &bytecode.instructions[i - 1] {
                            copies.insert(*dst, *src);
                        }
                    }
                }
                Instruction::LoadLocal(idx) => {
                    if let Some(&src) = copies.get(idx) {
                        bytecode.instructions[i] = Instruction::LoadLocal(src);
                        stats.instructions_modified += 1;
                    }
                }
                _ => {
                    // Any other instruction might use the variable, so clear copy info
                    // (conservative approach)
                }
            }
            i += 1;
        }

        Ok(stats)
    }
}

// Inline Small Functions
pub struct InlineSmallFunctions;

impl OptimizationPass for InlineSmallFunctions {
    fn name(&self) -> &str {
        "inline_small_functions"
    }

    fn run(&self, bytecode: &mut Bytecode) -> OmegaResult<OptimizationStats> {
        let stats = OptimizationStats::default();
        // Function inlining requires knowledge of function boundaries
        // This is a placeholder for the full implementation
        Ok(stats)
    }
}

// Helper functions
fn extract_number(constant: &Constant) -> Option<f64> {
    match constant {
        Constant::Integer(n) => Some(*n as f64),
        Constant::Float(f) => Some(*f),
        _ => None,
    }
}

fn constants_equal(a: &Constant, b: &Constant) -> bool {
    match (a, b) {
        (Constant::None, Constant::None) => true,
        (Constant::Bool(a), Constant::Bool(b)) => a == b,
        (Constant::Integer(a), Constant::Integer(b)) => a == b,
        (Constant::Float(a), Constant::Float(b)) => a == b,
        (Constant::String(a), Constant::String(b)) => a == b,
        (Constant::Char(a), Constant::Char(b)) => a == b,
        _ => false,
    }
}

fn is_jump_target(bytecode: &Bytecode, target: usize) -> bool {
    for instruction in &bytecode.instructions {
        match instruction {
            Instruction::Jump(t)
            | Instruction::JumpIfTrue(t)
            | Instruction::JumpIfFalse(t)
            | Instruction::JumpIfNone(t)
            | Instruction::JumpBack(t) => {
                if *t as usize == target {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}
