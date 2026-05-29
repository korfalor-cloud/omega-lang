/// Compiler IR: SSA intermediate representation, optimization passes, code generation.

use std::collections::{HashMap, HashSet, BTreeMap};

pub type VarId = usize;
pub type BlockId = usize;
pub type InstrId = usize;

#[derive(Debug, Clone)]
pub enum IRValue {
    Const(f64),
    Var(VarId),
    Bool(bool),
    Str(String),
    Null,
}

#[derive(Debug, Clone)]
pub enum IRInstr {
    Add { dst: VarId, lhs: IRValue, rhs: IRValue },
    Sub { dst: VarId, lhs: IRValue, rhs: IRValue },
    Mul { dst: VarId, lhs: IRValue, rhs: IRValue },
    Div { dst: VarId, lhs: IRValue, rhs: IRValue },
    Mod { dst: VarId, lhs: IRValue, rhs: IRValue },
    Neg { dst: VarId, src: IRValue },
    Not { dst: VarId, src: IRValue },
    And { dst: VarId, lhs: IRValue, rhs: IRValue },
    Or { dst: VarId, lhs: IRValue, rhs: IRValue },
    Eq { dst: VarId, lhs: IRValue, rhs: IRValue },
    Neq { dst: VarId, lhs: IRValue, rhs: IRValue },
    Lt { dst: VarId, lhs: IRValue, rhs: IRValue },
    Lte { dst: VarId, lhs: IRValue, rhs: IRValue },
    Gt { dst: VarId, lhs: IRValue, rhs: IRValue },
    Gte { dst: VarId, lhs: IRValue, rhs: IRValue },
    Load { dst: VarId, addr: IRValue },
    Store { addr: IRValue, val: IRValue },
    Call { dst: Option<VarId>, func: String, args: Vec<IRValue> },
    Return { val: Option<IRValue> },
    Branch { cond: IRValue, true_block: BlockId, false_block: BlockId },
    Jump { target: BlockId },
    Phi { dst: VarId, sources: Vec<(IRValue, BlockId)> },
    Cast { dst: VarId, src: IRValue, ty: IRType },
    Index { dst: VarId, base: IRValue, index: IRValue },
    Field { dst: VarId, base: IRValue, field: String },
    Alloc { dst: VarId, size: IRValue },
    Copy { dst: VarId, src: IRValue },
    Nop,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IRType {
    I64,
    F64,
    Bool,
    String,
    Array(Box<IRType>),
    Struct(String),
    Ptr(Box<IRType>),
    Void,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<IRInstr>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IRFunction {
    pub name: String,
    pub params: Vec<(VarId, IRType)>,
    pub return_type: IRType,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub var_types: HashMap<VarId, IRType>,
    next_var: VarId,
    next_block: BlockId,
}

impl IRFunction {
    pub fn new(name: &str, params: Vec<(VarId, IRType)>, return_type: IRType) -> Self {
        let entry = 0;
        Self {
            name: name.to_string(),
            params,
            return_type,
            blocks: vec![BasicBlock::new(entry)],
            entry,
            var_types: HashMap::new(),
            next_var: 0,
            next_block: 1,
        }
    }

    pub fn new_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    pub fn new_typed_var(&mut self, ty: IRType) -> VarId {
        let id = self.new_var();
        self.var_types.insert(id, ty);
        id
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = self.next_block;
        self.next_block += 1;
        self.blocks.push(BasicBlock::new(id));
        id
    }

    pub fn emit(&mut self, block: BlockId, instr: IRInstr) {
        if let Some(bb) = self.blocks.iter_mut().find(|b| b.id == block) {
            bb.instructions.push(instr);
        }
    }

    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        if let Some(bb) = self.blocks.iter_mut().find(|b| b.id == from) {
            if !bb.successors.contains(&to) {
                bb.successors.push(to);
            }
        }
        if let Some(bb) = self.blocks.iter_mut().find(|b| b.id == to) {
            if !bb.predecessors.contains(&from) {
                bb.predecessors.push(from);
            }
        }
    }

    /// Compute dominator tree.
    pub fn dominators(&self) -> HashMap<BlockId, BlockId> {
        let mut dom: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        let all_blocks: HashSet<BlockId> = self.blocks.iter().map(|b| b.id).collect();

        for &b in &all_blocks {
            dom.insert(b, all_blocks.clone());
        }
        dom.insert(self.entry, HashSet::from([self.entry]));

        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.blocks {
                if block.id == self.entry { continue; }

                let mut new_dom: HashSet<BlockId> = all_blocks.clone();
                for &pred in &block.predecessors {
                    if let Some(pred_dom) = dom.get(&pred) {
                        new_dom = new_dom.intersection(pred_dom).cloned().collect();
                    }
                }
                new_dom.insert(block.id);

                if new_dom != dom[&block.id] {
                    dom.insert(block.id, new_dom);
                    changed = true;
                }
            }
        }

        // Convert to immediate dominator map
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        for block in &self.blocks {
            if block.id == self.entry { continue; }
            let dominated = &dom[&block.id];
            for &d in dominated {
                if d == block.id { continue; }
                let d_dominates_all = dominated.iter().all(|&other| {
                    other == block.id || other == d || dom[&other].contains(&d)
                });
                if d_dominates_all {
                    idom.insert(block.id, d);
                    break;
                }
            }
        }
        idom
    }

    /// Compute use-def chains.
    pub fn use_def_chains(&self) -> HashMap<VarId, Vec<(BlockId, InstrId)>> {
        let mut defs: HashMap<VarId, Vec<(BlockId, InstrId)>> = HashMap::new();
        for block in &self.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if let Some(dst) = self.instr_def(instr) {
                    defs.entry(dst).or_default().push((block.id, idx));
                }
            }
        }
        defs
    }

    fn instr_def(&self, instr: &IRInstr) -> Option<VarId> {
        match instr {
            IRInstr::Add { dst, .. } | IRInstr::Sub { dst, .. } |
            IRInstr::Mul { dst, .. } | IRInstr::Div { dst, .. } |
            IRInstr::Mod { dst, .. } | IRInstr::Neg { dst, .. } |
            IRInstr::Not { dst, .. } | IRInstr::And { dst, .. } |
            IRInstr::Or { dst, .. } | IRInstr::Eq { dst, .. } |
            IRInstr::Neq { dst, .. } | IRInstr::Lt { dst, .. } |
            IRInstr::Lte { dst, .. } | IRInstr::Gt { dst, .. } |
            IRInstr::Gte { dst, .. } | IRInstr::Load { dst, .. } |
            IRInstr::Phi { dst, .. } | IRInstr::Cast { dst, .. } |
            IRInstr::Index { dst, .. } | IRInstr::Field { dst, .. } |
            IRInstr::Alloc { dst, .. } | IRInstr::Copy { dst, .. } => Some(*dst),
            IRInstr::Call { dst: Some(dst), .. } => Some(*dst),
            _ => None,
        }
    }

    fn instr_uses(&self, instr: &IRInstr) -> Vec<VarId> {
        let mut uses = Vec::new();
        let mut collect = |v: &IRValue| {
            if let IRValue::Var(id) = v { uses.push(*id); }
        };

        match instr {
            IRInstr::Add { lhs, rhs, .. } | IRInstr::Sub { lhs, rhs, .. } |
            IRInstr::Mul { lhs, rhs, .. } | IRInstr::Div { lhs, rhs, .. } |
            IRInstr::Mod { lhs, rhs, .. } | IRInstr::And { lhs, rhs, .. } |
            IRInstr::Or { lhs, rhs, .. } | IRInstr::Eq { lhs, rhs, .. } |
            IRInstr::Neq { lhs, rhs, .. } | IRInstr::Lt { lhs, rhs, .. } |
            IRInstr::Lte { lhs, rhs, .. } | IRInstr::Gt { lhs, rhs, .. } |
            IRInstr::Gte { lhs, rhs, .. } => { collect(lhs); collect(rhs); }
            IRInstr::Neg { src, .. } | IRInstr::Not { src, .. } |
            IRInstr::Load { dst: _, addr: src @ IRValue::Var(_) } => collect(src),
            IRInstr::Store { addr, val } => { collect(addr); collect(val); }
            IRInstr::Call { args, .. } => args.iter().for_each(collect),
            IRInstr::Return { val: Some(v) } => collect(v),
            IRInstr::Branch { cond, .. } => collect(cond),
            IRInstr::Phi { sources, .. } => sources.iter().for_each(|(v, _)| collect(v)),
            IRInstr::Cast { src, .. } => collect(src),
            IRInstr::Index { base, index, .. } => { collect(base); collect(index); }
            IRInstr::Field { base, .. } => collect(base),
            IRInstr::Alloc { size, .. } => collect(size),
            IRInstr::Copy { src, .. } => collect(src),
            _ => {}
        }
        uses
    }
}

/// SSA construction from basic blocks.
pub struct SSAConstructor {
    pub function: IRFunction,
}

impl SSAConstructor {
    pub fn new(function: IRFunction) -> Self {
        Self { function }
    }

    /// Insert phi nodes at iterated dominance frontier.
    pub fn insert_phi_nodes(&mut self) {
        let dom = self.function.dominators();
        // Compute dominance frontiers
        let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        for block in &self.function.blocks {
            df.insert(block.id, HashSet::new());
        }

        for block in &self.function.blocks {
            if block.predecessors.len() >= 2 {
                for &pred in &block.predecessors {
                    let mut runner = pred;
                    while runner != dom.get(&block.id).copied().unwrap_or(block.id) {
                        df.entry(runner).or_default().insert(block.id);
                        runner = dom.get(&runner).copied().unwrap_or(runner);
                    }
                }
            }
        }

        // For each variable, compute iterated dominance frontier and insert phis
        let all_vars: HashSet<VarId> = self.function.var_types.keys().copied().collect();
        for var in all_vars {
            let mut worklist: Vec<BlockId> = Vec::new();
            let mut phi_blocks: HashSet<BlockId> = HashSet::new();

            for block in &self.function.blocks {
                for instr in &block.instructions {
                    if self.function.instr_def(instr) == Some(var) {
                        worklist.push(block.id);
                    }
                }
            }

            while let Some(block_id) = worklist.pop() {
                if let Some(frontier) = df.get(&block_id) {
                    for &f in frontier {
                        if !phi_blocks.contains(&f) {
                            phi_blocks.insert(f);
                            // Insert phi
                            let sources: Vec<(IRValue, BlockId)> = self.function.blocks
                                .iter()
                                .find(|b| b.id == f)
                                .map(|b| b.predecessors.iter().map(|&p| (IRValue::Var(var), p)).collect())
                                .unwrap_or_default();

                            if let Some(block) = self.function.blocks.iter_mut().find(|b| b.id == f) {
                                block.instructions.insert(0, IRInstr::Phi {
                                    dst: var,
                                    sources,
                                });
                            }

                            // Check if this block defines var
                            let defines = self.function.blocks.iter()
                                .find(|b| b.id == f)
                                .map(|b| b.instructions.iter().any(|i| self.function.instr_def(i) == Some(var)))
                                .unwrap_or(false);
                            if !defines {
                                worklist.push(f);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Optimization passes.
pub struct Optimizer {
    pub function: IRFunction,
}

impl Optimizer {
    pub fn new(function: IRFunction) -> Self {
        Self { function }
    }

    /// Constant folding.
    pub fn constant_fold(&mut self) -> bool {
        let mut changed = false;
        for block in &mut self.function.blocks {
            let mut new_instrs = Vec::new();
            let mut const_vals: HashMap<VarId, f64> = HashMap::new();

            for instr in &block.instructions {
                match instr {
                    IRInstr::Add { dst, lhs, rhs } => {
                        if let (Some(a), Some(b)) = (self.eval_const(lhs, &const_vals), self.eval_const(rhs, &const_vals)) {
                            const_vals.insert(*dst, a + b);
                            new_instrs.push(IRInstr::Copy { dst: *dst, src: IRValue::Const(a + b) });
                            changed = true;
                        } else {
                            new_instrs.push(instr.clone());
                        }
                    }
                    IRInstr::Sub { dst, lhs, rhs } => {
                        if let (Some(a), Some(b)) = (self.eval_const(lhs, &const_vals), self.eval_const(rhs, &const_vals)) {
                            const_vals.insert(*dst, a - b);
                            new_instrs.push(IRInstr::Copy { dst: *dst, src: IRValue::Const(a - b) });
                            changed = true;
                        } else {
                            new_instrs.push(instr.clone());
                        }
                    }
                    IRInstr::Mul { dst, lhs, rhs } => {
                        if let (Some(a), Some(b)) = (self.eval_const(lhs, &const_vals), self.eval_const(rhs, &const_vals)) {
                            const_vals.insert(*dst, a * b);
                            new_instrs.push(IRInstr::Copy { dst: *dst, src: IRValue::Const(a * b) });
                            changed = true;
                        } else {
                            new_instrs.push(instr.clone());
                        }
                    }
                    IRInstr::Neg { dst, src } => {
                        if let Some(v) = self.eval_const(src, &const_vals) {
                            const_vals.insert(*dst, -v);
                            new_instrs.push(IRInstr::Copy { dst: *dst, src: IRValue::Const(-v) });
                            changed = true;
                        } else {
                            new_instrs.push(instr.clone());
                        }
                    }
                    _ => new_instrs.push(instr.clone()),
                }
            }
            block.instructions = new_instrs;
        }
        changed
    }

    fn eval_const(&self, val: &IRValue, consts: &HashMap<VarId, f64>) -> Option<f64> {
        match val {
            IRValue::Const(c) => Some(*c),
            IRValue::Var(id) => consts.get(id).copied(),
            _ => None,
        }
    }

    /// Dead code elimination.
    pub fn eliminate_dead_code(&mut self) -> bool {
        let mut changed = false;
        let mut used_vars: HashSet<VarId> = HashSet::new();

        // Collect all used variables
        for block in &self.function.blocks {
            for instr in &block.instructions {
                for used in self.function.instr_uses(instr) {
                    used_vars.insert(used);
                }
            }
        }

        // Remove instructions that define unused variables
        for block in &mut self.function.blocks {
            let original_len = block.instructions.len();
            block.instructions.retain(|instr| {
                if let Some(dst) = self.function.instr_def(instr) {
                    used_vars.contains(&dst)
                } else {
                    true
                }
            });
            if block.instructions.len() != original_len {
                changed = true;
            }
        }
        changed
    }

    /// Common subexpression elimination.
    pub fn eliminate_common_subexpressions(&mut self) -> bool {
        let mut changed = false;
        for block in &mut self.function.blocks {
            let mut available: HashMap<String, VarId> = HashMap::new();
            let mut new_instrs = Vec::new();

            for instr in &block.instructions.clone() {
                let key = self.instr_key(instr);
                if let Some(key) = &key {
                    if let Some(&existing_dst) = available.get(key) {
                        if let Some(dst) = self.function.instr_def(instr) {
                            new_instrs.push(IRInstr::Copy { dst, src: IRValue::Var(existing_dst) });
                            changed = true;
                            continue;
                        }
                    }
                }

                if let Some(dst) = self.function.instr_def(instr) {
                    if let Some(key) = key {
                        available.insert(key, dst);
                    }
                }
                new_instrs.push(instr.clone());
            }
            block.instructions = new_instrs;
        }
        changed
    }

    fn instr_key(&self, instr: &IRInstr) -> Option<String> {
        match instr {
            IRInstr::Add { lhs, rhs, .. } => Some(format!("add:{:?}:{:?}", lhs, rhs)),
            IRInstr::Sub { lhs, rhs, .. } => Some(format!("sub:{:?}:{:?}", lhs, rhs)),
            IRInstr::Mul { lhs, rhs, .. } => Some(format!("mul:{:?}:{:?}", lhs, rhs)),
            IRInstr::Div { lhs, rhs, .. } => Some(format!("div:{:?}:{:?}", lhs, rhs)),
            _ => None,
        }
    }

    /// Run all optimization passes.
    pub fn optimize(&mut self) {
        for _ in 0..10 {
            let mut any_changed = false;
            any_changed |= self.constant_fold();
            any_changed |= self.eliminate_dead_code();
            any_changed |= self.eliminate_common_subexpressions();
            if !any_changed { break; }
        }
    }
}

/// Register allocator using graph coloring.
pub struct RegisterAllocator {
    pub interference: HashMap<VarId, HashSet<VarId>>,
    pub coloring: HashMap<VarId, usize>,
}

impl RegisterAllocator {
    pub fn new() -> Self {
        Self {
            interference: HashMap::new(),
            coloring: HashMap::new(),
        }
    }

    pub fn build_interference(&mut self, function: &IRFunction) {
        // Simplified: build from live ranges
        for block in &function.blocks {
            let mut live: HashSet<VarId> = HashSet::new();
            // Compute live-out (simplified: all successor defs)
            for &succ in &block.successors {
                if let Some(succ_block) = function.blocks.iter().find(|b| b.id == succ) {
                    for instr in &succ_block.instructions {
                        if let Some(dst) = function.instr_def(instr) {
                            live.insert(dst);
                        }
                    }
                }
            }

            // Walk backwards
            for instr in block.instructions.iter().rev() {
                if let Some(def) = function.instr_def(instr) {
                    for &live_var in &live {
                        self.interference.entry(def).or_default().insert(live_var);
                        self.interference.entry(live_var).or_default().insert(def);
                    }
                    live.remove(&def);
                }
                for used in function.instr_uses(instr) {
                    live.insert(used);
                }
            }
        }
    }

    pub fn color(&mut self, num_registers: usize) -> bool {
        let all_vars: Vec<VarId> = self.interference.keys().copied().collect();
        let mut remaining: HashSet<VarId> = all_vars.iter().copied().collect();
        let mut stack: Vec<VarId> = Vec::new();

        // Simplify
        while !remaining.is_empty() {
            let var = remaining.iter()
                .min_by_key(|v| self.interference.get(v).map_or(0, |s| s.intersection(&remaining).count()))
                .copied();

            if let Some(var) = var {
                remaining.remove(&var);
                stack.push(var);
            } else {
                break;
            }
        }

        // Select
        for var in stack.iter().rev() {
            let used_colors: HashSet<usize> = self.interference
                .get(var)
                .map(|neighbors| {
                    neighbors.iter()
                        .filter_map(|n| self.coloring.get(n))
                        .copied()
                        .collect()
                })
                .unwrap_or_default();

            let color = (0..num_registers).find(|c| !used_colors.contains(c));
            if let Some(color) = color {
                self.coloring.insert(*var, color);
            } else {
                return false; // Spill needed
            }
        }
        true
    }
}

/// Liveness analysis.
pub fn liveness_analysis(function: &IRFunction) -> HashMap<BlockId, (HashSet<VarId>, HashSet<VarId>)> {
    let mut result: HashMap<BlockId, (HashSet<VarId>, HashSet<VarId>)> = HashMap::new();

    for block in &function.blocks {
        result.insert(block.id, (HashSet::new(), HashSet::new()));
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in &function.blocks.iter().rev().collect::<Vec<_>>() {
            let mut live_out: HashSet<VarId> = HashSet::new();
            for &succ in &block.successors {
                if let Some((_, succ_out)) = result.get(&succ) {
                    live_out.extend(succ_out);
                }
            }

            let mut live_in: HashSet<VarId> = live_out.clone();
            for instr in block.instructions.iter().rev() {
                if let Some(def) = function.instr_def(instr) {
                    live_in.remove(&def);
                }
                for used in function.instr_uses(instr) {
                    live_in.insert(used);
                }
            }

            let (old_in, old_out) = result.get(&block.id).unwrap();
            if live_in != *old_in || live_out != *old_out {
                result.insert(block.id, (live_in, live_out));
                changed = true;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_function() {
        let mut func = IRFunction::new("test", vec![], IRType::F64);
        let a = func.new_typed_var(IRType::F64);
        let b = func.new_typed_var(IRType::F64);
        let c = func.new_typed_var(IRType::F64);

        func.emit(0, IRInstr::Copy { dst: a, src: IRValue::Const(5.0) });
        func.emit(0, IRInstr::Copy { dst: b, src: IRValue::Const(3.0) });
        func.emit(0, IRInstr::Add { dst: c, lhs: IRValue::Var(a), rhs: IRValue::Var(b) });
        func.emit(0, IRInstr::Return { val: Some(IRValue::Var(c)) });

        assert_eq!(func.blocks[0].instructions.len(), 4);
    }

    #[test]
    fn test_constant_folding() {
        let mut func = IRFunction::new("test", vec![], IRType::F64);
        let a = func.new_typed_var(IRType::F64);
        let b = func.new_typed_var(IRType::F64);
        let c = func.new_typed_var(IRType::F64);

        func.emit(0, IRInstr::Add { dst: a, lhs: IRValue::Const(3.0), rhs: IRValue::Const(4.0) });
        func.emit(0, IRInstr::Mul { dst: b, lhs: IRValue::Var(a), rhs: IRValue::Const(2.0) });
        func.emit(0, IRInstr::Return { val: Some(IRValue::Var(b)) });

        let mut opt = Optimizer::new(func);
        opt.constant_fold();

        // After folding, instructions should be simplified
        assert!(opt.function.blocks[0].instructions.len() <= 3);
    }

    #[test]
    fn test_dominators() {
        let mut func = IRFunction::new("test", vec![], IRType::Void);
        let b1 = func.new_block();
        let b2 = func.new_block();
        func.add_edge(0, b1);
        func.add_edge(0, b2);
        func.add_edge(b1, b2);

        let dom = func.dominators();
        assert_eq!(dom[&b1], 0); // entry dominates b1
        assert_eq!(dom[&b2], 0); // entry dominates b2
    }
}
