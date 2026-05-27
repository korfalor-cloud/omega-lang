use omega_lang::ir::cfg::ControlFlowGraph;
use omega_lang::ir::ir_node::IrNode;

#[test]
fn test_empty_cfg() {
    let cfg = ControlFlowGraph::new();
    assert_eq!(cfg.num_blocks(), 0);
}

#[test]
fn test_sequential_cfg() {
    let nodes = vec![
        IrNode::ConstInteger(1),
        IrNode::ConstInteger(2),
        IrNode::Add(
            Box::new(IrNode::ConstInteger(1)),
            Box::new(IrNode::ConstInteger(2)),
        ),
    ];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    assert!(cfg.num_blocks() >= 1);
}

#[test]
fn test_if_cfg() {
    let nodes = vec![IrNode::If {
        condition: Box::new(IrNode::ConstBool(true)),
        then_branch: vec![IrNode::ConstInteger(1)],
        else_branch: vec![IrNode::ConstInteger(2)],
    }];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    // Should have entry, then, else, and merge blocks
    assert!(cfg.num_blocks() >= 3);
}

#[test]
fn test_while_cfg() {
    let nodes = vec![IrNode::While {
        condition: Box::new(IrNode::ConstBool(true)),
        body: vec![IrNode::ConstInteger(1)],
    }];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    // Should have entry, header, body, and exit blocks
    assert!(cfg.num_blocks() >= 3);
}

#[test]
fn test_cfg_entry() {
    let nodes = vec![IrNode::ConstInteger(42)];
    let cfg = ControlFlowGraph::from_ir(&nodes);

    let entry = cfg.block(cfg.entry()).unwrap();
    assert!(entry.is_entry());
    assert!(entry.predecessors.is_empty());
}

#[test]
fn test_cfg_dfs_order() {
    let nodes = vec![
        IrNode::ConstInteger(1),
        IrNode::ConstInteger(2),
    ];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    let order = cfg.dfs_order();
    assert!(!order.is_empty());
}

#[test]
fn test_cfg_post_order() {
    let nodes = vec![
        IrNode::ConstInteger(1),
        IrNode::ConstInteger(2),
    ];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    let order = cfg.post_order();
    assert!(!order.is_empty());
}

#[test]
fn test_cfg_reverse_post_order() {
    let nodes = vec![
        IrNode::ConstInteger(1),
        IrNode::ConstInteger(2),
    ];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    let order = cfg.reverse_post_order();
    assert!(!order.is_empty());
}

#[test]
fn test_cfg_dominators() {
    let nodes = vec![IrNode::ConstInteger(42)];
    let cfg = ControlFlowGraph::from_ir(&nodes);
    let dominators = cfg.compute_dominators();

    // Entry should dominate itself
    assert!(dominators.get(&cfg.entry()).is_some());
}

#[test]
fn test_cfg_cyclomatic_complexity() {
    let nodes = vec![IrNode::ConstInteger(42)];
    let cfg = ControlFlowGraph::from_ir(&nodes);
    let complexity = cfg.cyclomatic_complexity();

    // Simple program should have low complexity
    assert!(complexity >= 1);
}

#[test]
fn test_cfg_num_edges() {
    let nodes = vec![IrNode::ConstInteger(42)];
    let cfg = ControlFlowGraph::from_ir(&nodes);

    // At least one edge from entry to exit
    assert!(cfg.num_edges() >= 1);
}

#[test]
fn test_basic_block_terminator() {
    let nodes = vec![IrNode::Return(Some(Box::new(IrNode::ConstInteger(42))))];
    let cfg = ControlFlowGraph::from_ir(&nodes);

    let exit_blocks = cfg.exits();
    assert!(!exit_blocks.is_empty());
}

#[test]
fn test_loop_detection() {
    let nodes = vec![IrNode::While {
        condition: Box::new(IrNode::ConstBool(true)),
        body: vec![IrNode::ConstInteger(1)],
    }];

    let cfg = ControlFlowGraph::from_ir(&nodes);
    let loops = cfg.find_loops();

    // Should detect at least one loop
    assert!(!loops.is_empty());
}

#[test]
fn test_ir_node_is_pure() {
    assert!(IrNode::ConstInteger(42).is_pure());
    assert!(IrNode::ConstFloat(3.14).is_pure());
    assert!(IrNode::ConstBool(true).is_pure());
    assert!(IrNode::ConstString("hello".to_string()).is_pure());
    assert!(IrNode::ConstNone.is_pure());

    assert!(IrNode::Add(
        Box::new(IrNode::ConstInteger(1)),
        Box::new(IrNode::ConstInteger(2))
    )
    .is_pure());

    assert!(!IrNode::Print(vec![IrNode::ConstInteger(1)], true).is_pure());
}

#[test]
fn test_ir_node_children() {
    let add = IrNode::Add(
        Box::new(IrNode::ConstInteger(1)),
        Box::new(IrNode::ConstInteger(2)),
    );
    assert_eq!(add.children().len(), 2);

    let neg = IrNode::Neg(Box::new(IrNode::ConstInteger(1)));
    assert_eq!(neg.children().len(), 1);

    let arr = IrNode::Array(vec![
        IrNode::ConstInteger(1),
        IrNode::ConstInteger(2),
        IrNode::ConstInteger(3),
    ]);
    assert_eq!(arr.children().len(), 3);
}

#[test]
fn test_ir_node_display() {
    assert_eq!(IrNode::ConstInteger(42).to_string(), "42");
    assert_eq!(IrNode::ConstFloat(3.14).to_string(), "3.14");
    assert_eq!(IrNode::ConstBool(true).to_string(), "true");
    assert_eq!(
        IrNode::ConstString("hello".to_string()).to_string(),
        "\"hello\""
    );
    assert_eq!(IrNode::ConstNone.to_string(), "none");

    assert_eq!(
        IrNode::Add(
            Box::new(IrNode::ConstInteger(1)),
            Box::new(IrNode::ConstInteger(2))
        )
        .to_string(),
        "(1 + 2)"
    );
}
