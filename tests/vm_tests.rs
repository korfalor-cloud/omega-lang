use omega_lang::vm::stack::{Stack, Value};
use omega_lang::vm::heap::Heap;
use omega_lang::vm::machine::VirtualMachine;
use omega_lang::compiler::bytecode::{Bytecode, Instruction, Constant};

#[test]
fn test_stack_push_pop() {
    let mut stack = Stack::new();
    stack.push(Value::Integer(42)).unwrap();
    assert_eq!(stack.len(), 1);
    let val = stack.pop().unwrap();
    assert!(matches!(val, Value::Integer(42)));
}

#[test]
fn test_stack_multiple_values() {
    let mut stack = Stack::new();
    stack.push(Value::Integer(1)).unwrap();
    stack.push(Value::Integer(2)).unwrap();
    stack.push(Value::Integer(3)).unwrap();
    assert_eq!(stack.len(), 3);
    assert!(matches!(stack.pop().unwrap(), Value::Integer(3)));
    assert!(matches!(stack.pop().unwrap(), Value::Integer(2)));
    assert!(matches!(stack.pop().unwrap(), Value::Integer(1)));
}

#[test]
fn test_stack_dup() {
    let mut stack = Stack::new();
    stack.push(Value::Integer(42)).unwrap();
    stack.dup().unwrap();
    assert_eq!(stack.len(), 2);
    assert!(matches!(stack.pop().unwrap(), Value::Integer(42)));
    assert!(matches!(stack.pop().unwrap(), Value::Integer(42)));
}

#[test]
fn test_stack_swap() {
    let mut stack = Stack::new();
    stack.push(Value::Integer(1)).unwrap();
    stack.push(Value::Integer(2)).unwrap();
    stack.swap().unwrap();
    assert!(matches!(stack.pop().unwrap(), Value::Integer(1)));
    assert!(matches!(stack.pop().unwrap(), Value::Integer(2)));
}

#[test]
fn test_value_add_integers() {
    let a = Value::Integer(10);
    let b = Value::Integer(20);
    let result = a.add(&b).unwrap();
    assert!(matches!(result, Value::Integer(30)));
}

#[test]
fn test_value_add_floats() {
    let a = Value::Float(1.5);
    let b = Value::Float(2.5);
    let result = a.add(&b).unwrap();
    assert!(matches!(result, Value::Float(f) if (f - 4.0).abs() < 0.001));
}

#[test]
fn test_value_add_strings() {
    let a = Value::String("hello".to_string());
    let b = Value::String(" world".to_string());
    let result = a.add(&b).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "hello world"));
}

#[test]
fn test_value_sub() {
    let a = Value::Integer(30);
    let b = Value::Integer(10);
    let result = a.sub(&b).unwrap();
    assert!(matches!(result, Value::Integer(20)));
}

#[test]
fn test_value_mul() {
    let a = Value::Integer(5);
    let b = Value::Integer(6);
    let result = a.mul(&b).unwrap();
    assert!(matches!(result, Value::Integer(30)));
}

#[test]
fn test_value_div() {
    let a = Value::Integer(30);
    let b = Value::Integer(5);
    let result = a.div(&b).unwrap();
    assert!(matches!(result, Value::Integer(6)));
}

#[test]
fn test_value_modulo() {
    let a = Value::Integer(17);
    let b = Value::Integer(5);
    let result = a.modulo(&b).unwrap();
    assert!(matches!(result, Value::Integer(2)));
}

#[test]
fn test_value_neg() {
    let a = Value::Integer(42);
    let result = a.neg().unwrap();
    assert!(matches!(result, Value::Integer(-42)));
}

#[test]
fn test_value_eq() {
    let a = Value::Integer(42);
    let b = Value::Integer(42);
    assert!(matches!(a.eq(&b), Value::Bool(true)));

    let c = Value::Integer(100);
    assert!(matches!(a.eq(&c), Value::Bool(false)));
}

#[test]
fn test_value_ne() {
    let a = Value::Integer(42);
    let b = Value::Integer(100);
    assert!(matches!(a.ne(&b), Value::Bool(true)));
}

#[test]
fn test_value_lt() {
    let a = Value::Integer(10);
    let b = Value::Integer(20);
    assert!(matches!(a.lt(&b).unwrap(), Value::Bool(true)));
    assert!(matches!(b.lt(&a).unwrap(), Value::Bool(false)));
}

#[test]
fn test_value_gt() {
    let a = Value::Integer(20);
    let b = Value::Integer(10);
    assert!(matches!(a.gt(&b).unwrap(), Value::Bool(true)));
    assert!(matches!(b.gt(&a).unwrap(), Value::Bool(false)));
}

#[test]
fn test_value_le() {
    let a = Value::Integer(10);
    let b = Value::Integer(10);
    assert!(matches!(a.le(&b).unwrap(), Value::Bool(true)));
}

#[test]
fn test_value_ge() {
    let a = Value::Integer(10);
    let b = Value::Integer(10);
    assert!(matches!(a.ge(&b).unwrap(), Value::Bool(true)));
}

#[test]
fn test_value_bit_and() {
    let a = Value::Integer(0b1100);
    let b = Value::Integer(0b1010);
    let result = a.bit_and(&b).unwrap();
    assert!(matches!(result, Value::Integer(0b1000)));
}

#[test]
fn test_value_bit_or() {
    let a = Value::Integer(0b1100);
    let b = Value::Integer(0b1010);
    let result = a.bit_or(&b).unwrap();
    assert!(matches!(result, Value::Integer(0b1110)));
}

#[test]
fn test_value_bit_xor() {
    let a = Value::Integer(0b1100);
    let b = Value::Integer(0b1010);
    let result = a.bit_xor(&b).unwrap();
    assert!(matches!(result, Value::Integer(0b0110)));
}

#[test]
fn test_value_bit_not() {
    let a = Value::Integer(0);
    let result = a.bit_not().unwrap();
    assert!(matches!(result, Value::Integer(-1)));
}

#[test]
fn test_value_shl() {
    let a = Value::Integer(1);
    let b = Value::Integer(4);
    let result = a.shl(&b).unwrap();
    assert!(matches!(result, Value::Integer(16)));
}

#[test]
fn test_value_shr() {
    let a = Value::Integer(16);
    let b = Value::Integer(2);
    let result = a.shr(&b).unwrap();
    assert!(matches!(result, Value::Integer(4)));
}

#[test]
fn test_value_is_truthy() {
    assert!(Value::Bool(true).is_truthy());
    assert!(!Value::Bool(false).is_truthy());
    assert!(Value::Integer(1).is_truthy());
    assert!(!Value::Integer(0).is_truthy());
    assert!(Value::String("hello".to_string()).is_truthy());
    assert!(!Value::String("".to_string()).is_truthy());
    assert!(!Value::None.is_truthy());
}

#[test]
fn test_value_not() {
    let a = Value::Bool(true);
    assert!(matches!(a.not(), Value::Bool(false)));

    let b = Value::Bool(false);
    assert!(matches!(b.not(), Value::Bool(true)));
}

#[test]
fn test_value_type_name() {
    assert_eq!(Value::Integer(42).type_name(), "Integer");
    assert_eq!(Value::Float(3.14).type_name(), "Float");
    assert_eq!(Value::Bool(true).type_name(), "Bool");
    assert_eq!(Value::String("hello".to_string()).type_name(), "String");
    assert_eq!(Value::None.type_name(), "None");
}

#[test]
fn test_value_format_display() {
    assert_eq!(Value::Integer(42).format_display(), "42");
    assert_eq!(Value::Float(3.14).format_display(), "3.14");
    assert_eq!(Value::Bool(true).format_display(), "true");
    assert_eq!(Value::Bool(false).format_display(), "false");
    assert_eq!(Value::String("hello".to_string()).format_display(), "hello");
    assert_eq!(Value::None.format_display(), "none");
}

#[test]
fn test_array_operations() {
    let arr = vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
    ];
    let val = Value::Array(arr);
    assert_eq!(val.type_name(), "Array");
}

#[test]
fn test_tuple_operations() {
    let tuple = vec![
        Value::Integer(1),
        Value::String("hello".to_string()),
        Value::Bool(true),
    ];
    let val = Value::Tuple(tuple);
    assert_eq!(val.type_name(), "Tuple");
}

#[test]
fn test_map_operations() {
    let map = vec![
        (Value::String("key".to_string()), Value::Integer(42)),
    ];
    let val = Value::Map(map);
    assert_eq!(val.type_name(), "Map");
}

#[test]
fn test_heap_allocate() {
    let mut heap = Heap::new();
    let idx = heap.allocate(Value::Integer(42));
    assert_eq!(heap.allocated_count(), 1);
}

#[test]
fn test_heap_gc() {
    let mut heap = Heap::new();
    heap.allocate(Value::Integer(1));
    heap.allocate(Value::Integer(2));
    heap.allocate(Value::Integer(3));
    assert_eq!(heap.allocated_count(), 3);

    let roots = vec![Value::Integer(1)];
    heap.gc(&roots);
    // After GC, unreachable objects should be freed
}

#[test]
fn test_vm_basic() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(42)));
}

#[test]
fn test_vm_add() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Add,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(30)));
}

#[test]
fn test_vm_sub() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(30)),
            Instruction::Push(Constant::Integer(10)),
            Instruction::Sub,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(20)));
}

#[test]
fn test_vm_mul() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(5)),
            Instruction::Push(Constant::Integer(6)),
            Instruction::Mul,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(30)));
}

#[test]
fn test_vm_div() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(30)),
            Instruction::Push(Constant::Integer(5)),
            Instruction::Div,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(6)));
}

#[test]
fn test_vm_comparison() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Lt,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Bool(true)));
}

#[test]
fn test_vm_logical_and() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Bool(true)),
            Instruction::Push(Constant::Bool(false)),
            Instruction::And,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Bool(false)));
}

#[test]
fn test_vm_logical_or() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Bool(true)),
            Instruction::Push(Constant::Bool(false)),
            Instruction::Or,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Bool(true)));
}

#[test]
fn test_vm_not() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Bool(true)),
            Instruction::Not,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Bool(false)));
}

#[test]
fn test_vm_neg() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Neg,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(-42)));
}

#[test]
fn test_vm_dup_pop() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Dup,
            Instruction::Add,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(84)));
}

#[test]
fn test_vm_swap() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(10)),
            Instruction::Push(Constant::Integer(20)),
            Instruction::Swap,
            Instruction::Sub,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(10)));
}

#[test]
fn test_vm_string_concat() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::String("hello".to_string())),
            Instruction::Push(Constant::String(" world".to_string())),
            Instruction::StringConcat,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "hello world"));
}

#[test]
fn test_vm_make_array() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(1)),
            Instruction::Push(Constant::Integer(2)),
            Instruction::Push(Constant::Integer(3)),
            Instruction::MakeArray(3),
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    match result {
        Value::Array(arr) => assert_eq!(arr.len(), 3),
        _ => panic!("Expected Array"),
    }
}

#[test]
fn test_vm_make_tuple() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(1)),
            Instruction::Push(Constant::String("hello".to_string())),
            Instruction::Push(Constant::Bool(true)),
            Instruction::MakeTuple(3),
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    match result {
        Value::Tuple(t) => assert_eq!(t.len(), 3),
        _ => panic!("Expected Tuple"),
    }
}

#[test]
fn test_vm_type_of() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::TypeOf,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "Integer"));
}

#[test]
fn test_vm_halt() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(1)),
            Instruction::Halt,
            Instruction::Push(Constant::Integer(2)), // Should not execute
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(1)));
}

#[test]
fn test_vm_nop() {
    let mut vm = VirtualMachine::new();
    let chunk = Bytecode {
        instructions: vec![
            Instruction::Push(Constant::Integer(42)),
            Instruction::Nop,
        ],
        constants: vec![],
    };
    let result = vm.run(&[chunk]).unwrap();
    assert!(matches!(result, Value::Integer(42)));
}
