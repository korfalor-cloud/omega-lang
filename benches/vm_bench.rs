use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use omega_lang::vm::stack::{Stack, Value};
use omega_lang::vm::machine::VirtualMachine;
use omega_lang::compiler::bytecode::{Bytecode, Instruction, Constant};

fn bench_stack_push_pop(c: &mut Criterion) {
    c.bench_function("stack_push_pop", |b| {
        b.iter(|| {
            let mut stack = Stack::new();
            for i in 0..100 {
                stack.push(Value::Integer(i)).unwrap();
            }
            for _ in 0..100 {
                stack.pop().unwrap();
            }
        })
    });
}

fn bench_value_add_integers(c: &mut Criterion) {
    c.bench_function("value_add_integers", |b| {
        b.iter(|| {
            let a = Value::Integer(10);
            let b = Value::Integer(20);
            black_box(a.add(&b).unwrap())
        })
    });
}

fn bench_value_add_floats(c: &mut Criterion) {
    c.bench_function("value_add_floats", |b| {
        b.iter(|| {
            let a = Value::Float(1.5);
            let b = Value::Float(2.5);
            black_box(a.add(&b).unwrap())
        })
    });
}

fn bench_value_add_strings(c: &mut Criterion) {
    c.bench_function("value_add_strings", |b| {
        b.iter(|| {
            let a = Value::String("hello".to_string());
            let b = Value::String(" world".to_string());
            black_box(a.add(&b).unwrap())
        })
    });
}

fn bench_value_comparison(c: &mut Criterion) {
    c.bench_function("value_comparison", |b| {
        b.iter(|| {
            let a = Value::Integer(10);
            let b = Value::Integer(20);
            black_box(a.lt(&b).unwrap())
        })
    });
}

fn bench_vm_push(c: &mut Criterion) {
    c.bench_function("vm_push", |b| {
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            let chunk = Bytecode {
                instructions: vec![
                    Instruction::Push(Constant::Integer(42)),
                ],
                constants: vec![],
            };
            black_box(vm.run(&[chunk]).unwrap())
        })
    });
}

fn bench_vm_add(c: &mut Criterion) {
    c.bench_function("vm_add", |b| {
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            let chunk = Bytecode {
                instructions: vec![
                    Instruction::Push(Constant::Integer(10)),
                    Instruction::Push(Constant::Integer(20)),
                    Instruction::Add,
                ],
                constants: vec![],
            };
            black_box(vm.run(&[chunk]).unwrap())
        })
    });
}

fn bench_vm_loop(c: &mut Criterion) {
    c.bench_function("vm_loop", |b| {
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            let chunk = Bytecode {
                instructions: vec![
                    Instruction::Push(Constant::Integer(0)),  // counter
                    Instruction::Dup,                         // dup counter
                    Instruction::Push(Constant::Integer(100)), // limit
                    Instruction::Lt,                          // counter < limit
                    Instruction::JumpIfFalse(10),             // exit if false
                    Instruction::Push(Constant::Integer(1)),  // increment
                    Instruction::Add,                         // counter + 1
                    Instruction::Jump(1),                     // loop back
                    Instruction::Halt,
                ],
                constants: vec![],
            };
            black_box(vm.run(&[chunk]).unwrap())
        })
    });
}

fn bench_vm_make_array(c: &mut Criterion) {
    c.bench_function("vm_make_array", |b| {
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            let chunk = Bytecode {
                instructions: vec![
                    Instruction::Push(Constant::Integer(1)),
                    Instruction::Push(Constant::Integer(2)),
                    Instruction::Push(Constant::Integer(3)),
                    Instruction::Push(Constant::Integer(4)),
                    Instruction::Push(Constant::Integer(5)),
                    Instruction::MakeArray(5),
                ],
                constants: vec![],
            };
            black_box(vm.run(&[chunk]).unwrap())
        })
    });
}

fn bench_vm_string_concat(c: &mut Criterion) {
    c.bench_function("vm_string_concat", |b| {
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            let chunk = Bytecode {
                instructions: vec![
                    Instruction::Push(Constant::String("hello".to_string())),
                    Instruction::Push(Constant::String(" ".to_string())),
                    Instruction::StringConcat,
                    Instruction::Push(Constant::String("world".to_string())),
                    Instruction::StringConcat,
                ],
                constants: vec![],
            };
            black_box(vm.run(&[chunk]).unwrap())
        })
    });
}

fn bench_value_arithmetic_chain(c: &mut Criterion) {
    c.bench_function("value_arithmetic_chain", |b| {
        b.iter(|| {
            let mut result = Value::Integer(0);
            for i in 1..=100 {
                result = result.add(&Value::Integer(i)).unwrap();
            }
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    bench_stack_push_pop,
    bench_value_add_integers,
    bench_value_add_floats,
    bench_value_add_strings,
    bench_value_comparison,
    bench_vm_push,
    bench_vm_add,
    bench_vm_loop,
    bench_vm_make_array,
    bench_vm_string_concat,
    bench_value_arithmetic_chain,
);
criterion_main!(benches);
