use std::collections::{HashMap, HashSet, VecDeque};
use super::ir_node::IrNode;

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub instructions: Vec<IrNode>,
    pub successors: Vec<usize>,
    pub predecessors: Vec<usize>,
}

impl BasicBlock {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    pub fn is_entry(&self) -> bool {
        self.predecessors.is_empty()
    }

    pub fn is_exit(&self) -> bool {
        self.successors.is_empty()
    }

    pub fn terminator(&self) -> Option<&IrNode> {
        self.instructions.last()
    }
}

#[derive(Debug)]
pub struct ControlFlowGraph {
    blocks: Vec<BasicBlock>,
    entry: usize,
    exits: Vec<usize>,
}

impl ControlFlowGraph {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            entry: 0,
            exits: Vec::new(),
        }
    }

    pub fn from_ir(nodes: &[IrNode]) -> Self {
        let mut cfg = Self::new();
        let entry = cfg.new_block();
        cfg.entry = entry;

        let mut current = entry;
        for node in nodes {
            match node {
                IrNode::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    let then_block = cfg.new_block();
                    let else_block = cfg.new_block();
                    let merge_block = cfg.new_block();

                    cfg.add_edge(current, then_block);
                    cfg.add_edge(current, else_block);

                    // Then branch
                    for stmt in then_branch {
                        cfg.blocks[then_block].instructions.push(stmt.clone());
                    }
                    cfg.add_edge(then_block, merge_block);

                    // Else branch
                    for stmt in else_branch {
                        cfg.blocks[else_block].instructions.push(stmt.clone());
                    }
                    cfg.add_edge(else_block, merge_block);

                    current = merge_block;
                }
                IrNode::While { condition, body } => {
                    let header = cfg.new_block();
                    let loop_body = cfg.new_block();
                    let exit = cfg.new_block();

                    cfg.add_edge(current, header);
                    cfg.add_edge(header, loop_body);
                    cfg.add_edge(header, exit);
                    cfg.add_edge(loop_body, header);

                    for stmt in body {
                        cfg.blocks[loop_body].instructions.push(stmt.clone());
                    }

                    current = exit;
                }
                IrNode::Break => {
                    // Will be connected to loop exit
                    cfg.blocks[current].instructions.push(node.clone());
                }
                IrNode::Continue => {
                    // Will be connected to loop header
                    cfg.blocks[current].instructions.push(node.clone());
                }
                IrNode::Return(_) => {
                    cfg.blocks[current].instructions.push(node.clone());
                    cfg.exits.push(current);
                    let new_block = cfg.new_block();
                    current = new_block;
                }
                _ => {
                    cfg.blocks[current].instructions.push(node.clone());
                }
            }
        }

        cfg.exits.push(current);
        cfg
    }

    fn new_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(BasicBlock::new(id));
        id
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        self.blocks[from].successors.push(to);
        self.blocks[to].predecessors.push(from);
    }

    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    pub fn entry(&self) -> usize {
        self.entry
    }

    pub fn exits(&self) -> &[usize] {
        &self.exits
    }

    pub fn block(&self, id: usize) -> Option<&BasicBlock> {
        self.blocks.get(id)
    }

    pub fn block_mut(&mut self, id: usize) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(id)
    }

    // Dominance tree
    pub fn compute_dominators(&self) -> HashMap<usize, HashSet<usize>> {
        let mut dominators: HashMap<usize, HashSet<usize>> = HashMap::new();

        // Initialize
        for block in &self.blocks {
            let mut dom = HashSet::new();
            for b in &self.blocks {
                dom.insert(b.id);
            }
            dominators.insert(block.id, dom);
        }

        // Entry dominates itself
        if let Some(entry_dom) = dominators.get_mut(&self.entry) {
            entry_dom.clear();
            entry_dom.insert(self.entry);
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.blocks {
                if block.id == self.entry {
                    continue;
                }

                let mut new_dom: Option<HashSet<usize>> = None;
                for &pred in &block.predecessors {
                    if let Some(pred_dom) = dominators.get(&pred) {
                        let mut intersection = pred_dom.clone();
                        intersection.insert(block.id);
                        new_dom = Some(match new_dom {
                            Some(existing) => existing.intersection(&intersection).cloned().collect(),
                            None => intersection,
                        });
                    }
                }

                if let Some(new_dom) = new_dom {
                    if dominators.get(&block.id) != Some(&new_dom) {
                        dominators.insert(block.id, new_dom);
                        changed = true;
                    }
                }
            }
        }

        dominators
    }

    // Immediate dominators
    pub fn compute_immediate_dominators(&self) -> HashMap<usize, usize> {
        let dominators = self.compute_dominators();
        let mut idom: HashMap<usize, usize> = HashMap::new();

        for block in &self.blocks {
            if block.id == self.entry {
                continue;
            }

            if let Some(dom_set) = dominators.get(&block.id) {
                // Immediate dominator is the dominator closest to the block
                let candidates: Vec<usize> = dom_set
                    .iter()
                    .filter(|&&d| d != block.id)
                    .copied()
                    .collect();

                for &candidate in &candidates {
                    let dominated_by_candidate = dominators
                        .get(&candidate)
                        .map(|d| d.contains(&block.id))
                        .unwrap_or(false);

                    if !dominated_by_candidate {
                        idom.insert(block.id, candidate);
                        break;
                    }
                }
            }
        }

        idom
    }

    // Depth-first search ordering
    pub fn dfs_order(&self) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.dfs(self.entry, &mut visited, &mut order);
        order
    }

    fn dfs(&self, block_id: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if visited.contains(&block_id) {
            return;
        }
        visited.insert(block_id);
        order.push(block_id);

        if let Some(block) = self.block(block_id) {
            for &succ in &block.successors {
                self.dfs(succ, visited, order);
            }
        }
    }

    // Post-order traversal
    pub fn post_order(&self) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.post_order_dfs(self.entry, &mut visited, &mut order);
        order
    }

    fn post_order_dfs(
        &self,
        block_id: usize,
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) {
        if visited.contains(&block_id) {
            return;
        }
        visited.insert(block_id);

        if let Some(block) = self.block(block_id) {
            for &succ in &block.successors {
                self.post_order_dfs(succ, visited, order);
            }
        }

        order.push(block_id);
    }

    // Reverse post-order (good for dataflow analysis)
    pub fn reverse_post_order(&self) -> Vec<usize> {
        let mut order = self.post_order();
        order.reverse();
        order
    }

    // Check if the graph is reducible
    pub fn is_reducible(&self) -> bool {
        // A CFG is reducible if all back edges go to a dominator
        let dominators = self.compute_dominators();

        for block in &self.blocks {
            for &succ in &block.successors {
                // Check if this is a back edge
                if self.is_back_edge(block.id, succ) {
                    if let Some(dom_set) = dominators.get(&block.id) {
                        if !dom_set.contains(&succ) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    fn is_back_edge(&self, from: usize, to: usize) -> bool {
        // A back edge goes to an ancestor in the DFS tree
        let dfs_order = self.dfs_order();
        let from_pos = dfs_order.iter().position(|&x| x == from);
        let to_pos = dfs_order.iter().position(|&x| x == to);

        match (from_pos, to_pos) {
            (Some(f), Some(t)) => t <= f,
            _ => false,
        }
    }

    // Loop detection
    pub fn find_loops(&self) -> Vec<LoopInfo> {
        let mut loops = Vec::new();
        let dominators = self.compute_dominators();

        for block in &self.blocks {
            for &succ in &block.successors {
                if self.is_back_edge(block.id, succ) {
                    // Found a loop with header 'succ'
                    let mut loop_blocks = HashSet::new();
                    loop_blocks.insert(succ);

                    if succ != block.id {
                        loop_blocks.insert(block.id);
                        // Add all blocks that can reach the tail without going through the header
                        self.find_loop_blocks(succ, block.id, &mut loop_blocks);
                    }

                    loops.push(LoopInfo {
                        header: succ,
                        back_edges: vec![(block.id, succ)],
                        blocks: loop_blocks,
                    });
                }
            }
        }

        loops
    }

    fn find_loop_blocks(
        &self,
        header: usize,
        tail: usize,
        loop_blocks: &mut HashSet<usize>,
    ) {
        let mut worklist = VecDeque::new();
        worklist.push_back(tail);

        while let Some(current) = worklist.pop_front() {
            if current == header {
                continue;
            }

            if let Some(block) = self.block(current) {
                for &pred in &block.predecessors {
                    if !loop_blocks.contains(&pred) {
                        loop_blocks.insert(pred);
                        worklist.push_back(pred);
                    }
                }
            }
        }
    }

    // Number of blocks and edges
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_edges(&self) -> usize {
        self.blocks.iter().map(|b| b.successors.len()).sum()
    }

    // Cyclomatic complexity
    pub fn cyclomatic_complexity(&self) -> usize {
        self.num_edges() - self.num_blocks() + 2
    }
}

#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub header: usize,
    pub back_edges: Vec<(usize, usize)>,
    pub blocks: HashSet<usize>,
}

impl LoopInfo {
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn contains(&self, block_id: usize) -> bool {
        self.blocks.contains(&block_id)
    }
}
