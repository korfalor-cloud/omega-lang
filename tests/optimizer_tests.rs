use omega_lang::optimizer::Optimizer;
use omega_lang::compiler::bytecode::{Bytecode, Instruction, Constant};

#[test]
fn test_constant_folding_add() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Add,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Float(f)) if (*f - 30.0).abs() < 0.001
    ));
}

#[test]
fn test_constant_folding_sub() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(30)),
            Instruction::Push(Constant::Integer(10)),
            Instruction::Sub,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Float(f)) if (*f - 20.0).abs() < 0.001
    ));
}

#[test]
fn test_constant_folding_mul() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(5)),
            Instruction::Push(Constant::Integer(6)),
            Instruction::Mul,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Float(f)) if (*f - 30.0).abs() < 0.001
    ));
}

#[test]
fn test_constant_folding_div() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(30)),
            Instruction::Push(Constant::Integer(5)),
            Instruction::Div,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Float(f)) if (*f - 6.0).abs() < 0.001
    ));
}

#[test]
fn test_dead_code_elimination_push_pop() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Pop,
            Instruction::Push(Constant::Integer(100)),
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(100))
    ));
}

#[test]
fn test_peephole_zero_add() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Push(Constant::Integer(0)),
            Instruction::Add,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(42))
    ));
}

#[test]
fn test_peephole_one_mul() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Push(Constant::Integer(1)),
            Instruction::Mul,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(42))
    ));
}

#[test]
fn test_peephole_zero_mul() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Push(Constant::Integer(0)),
            Instruction::Mul,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(0))
    ));
}

#[test]
fn test_peephole_double_neg() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Neg,
            Instruction::Neg,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(42))
    ));
}

#[test]
fn test_peephole_double_not() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Bool(true)),
            Instruction::Not,
            Instruction::Not,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Bool(true))
    ));
}

#[test]
fn test_strength_reduction_mul_2() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Push(Constant::Integer(2)),
            Instruction::Mul,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(1))
    ));
    assert!(matches!(&bytecode.instructions[1], Instruction::Shl));
}

#[test]
fn test_strength_reduction_div_2() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Push(Constant::Integer(2)),
            Instruction::Div,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert!(matches!(
        &bytecode.instructions[0],
        Instruction::Push(Constant::Integer(1))
    ));
    assert!(matches!(&bytecode.instructions[1], Instruction::Shr));
}

#[test]
fn test_no_optimization_needed() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Halt,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 2);
    assert_eq!(stats.total_changes(), 0);
}

#[test]
fn test_multiple_optimizations() {
    let optimizer = Optimizer::new();
    let mut bytecode = Bytecode {
        instructions: vec![
            // First: constant fold
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Add,
            // Second: dead code
            Instruction::Push(Constant::Integer(0)),
            Instruction::Pop,
            // Third: peephole
            Instruction::Push(Constant::Integer(0)),
            Instruction::Add,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert!(stats.total_changes() > 0);
}

#[test]
fn test_optimizer_debug_mode() {
    let optimizer = Optimizer::new().with_debug();
    let mut bytecode = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Add,
        ],
        constants: vec![],
    };

    let stats = optimizer.optimize(&mut bytecode).unwrap();
    assert_eq!(bytecode.instructions.len(), 1);
}
