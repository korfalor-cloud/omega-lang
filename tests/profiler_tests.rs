use omega_lang::profiler::Profiler;
use std::time::Duration;

#[test]
fn test_profiler_new() {
    let profiler = Profiler::new();
    assert!(profiler.is_enabled());
    assert_eq!(profiler.total_instructions(), 0);
}

#[test]
fn test_profiler_enable_disable() {
    let mut profiler = Profiler::new();
    assert!(profiler.is_enabled());

    profiler.disable();
    assert!(!profiler.is_enabled());

    profiler.enable();
    assert!(profiler.is_enabled());
}

#[test]
fn test_profiler_record_instruction() {
    let mut profiler = Profiler::new();

    profiler.record_instruction("Add");
    profiler.record_instruction("Add");
    profiler.record_instruction("Sub");

    assert_eq!(profiler.total_instructions(), 3);
    assert_eq!(profiler.instruction_counts().get("Add"), Some(&2));
    assert_eq!(profiler.instruction_counts().get("Sub"), Some(&1));
}

#[test]
fn test_profiler_disabled_no_record() {
    let mut profiler = Profiler::new();
    profiler.disable();

    profiler.record_instruction("Add");
    assert_eq!(profiler.total_instructions(), 0);
}

#[test]
fn test_profiler_enter_exit_function() {
    let mut profiler = Profiler::new();

    profiler.enter_function("test_fn");
    // Simulate some work
    std::thread::sleep(Duration::from_millis(10));
    profiler.exit_function();

    let entry = profiler.get_entry("test_fn");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().call_count, 1);
}

#[test]
fn test_profiler_multiple_calls() {
    let mut profiler = Profiler::new();

    for _ in 0..5 {
        profiler.enter_function("test_fn");
        profiler.exit_function();
    }

    let entry = profiler.get_entry("test_fn").unwrap();
    assert_eq!(entry.call_count, 5);
}

#[test]
fn test_profiler_nested_functions() {
    let mut profiler = Profiler::new();

    profiler.enter_function("outer");
    profiler.enter_function("inner");
    profiler.exit_function();
    profiler.exit_function();

    assert!(profiler.get_entry("outer").is_some());
    assert!(profiler.get_entry("inner").is_some());
}

#[test]
fn test_profiler_memory_snapshot() {
    let mut profiler = Profiler::new();

    profiler.take_memory_snapshot(1024, 512, 10);
    profiler.take_memory_snapshot(2048, 1024, 20);

    assert_eq!(profiler.memory_snapshots().len(), 2);
}

#[test]
fn test_profiler_report() {
    let mut profiler = Profiler::new();

    profiler.enter_function("fast_fn");
    profiler.exit_function();

    profiler.enter_function("slow_fn");
    std::thread::sleep(Duration::from_millis(50));
    profiler.exit_function();

    let report = profiler.report();
    assert!(report.total_runtime_ms > 0.0);
}

#[test]
fn test_profiler_reset() {
    let mut profiler = Profiler::new();

    profiler.enter_function("test_fn");
    profiler.exit_function();
    profiler.record_instruction("Add");

    profiler.reset();

    assert_eq!(profiler.total_instructions(), 0);
    assert!(profiler.get_entry("test_fn").is_none());
}

#[test]
fn test_profiler_report_format() {
    let mut profiler = Profiler::new();

    profiler.enter_function("fn_a");
    profiler.exit_function();
    profiler.enter_function("fn_b");
    profiler.exit_function();

    let report = profiler.report();
    let formatted = format!("{}", report);
    assert!(formatted.contains("Profile Report"));
}
