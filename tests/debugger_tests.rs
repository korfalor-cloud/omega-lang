use omega_lang::debugger::{Debugger, DebugCommand, DebuggerSession};

#[test]
fn test_debugger_new() {
    let debugger = Debugger::new();
    assert_eq!(debugger.list_breakpoints().len(), 0);
}

#[test]
fn test_debugger_add_breakpoint() {
    let mut debugger = Debugger::new();
    let id = debugger.add_breakpoint(100);
    assert_eq!(id, 1);
    assert_eq!(debugger.list_breakpoints().len(), 1);
}

#[test]
fn test_debugger_multiple_breakpoints() {
    let mut debugger = Debugger::new();
    debugger.add_breakpoint(100);
    debugger.add_breakpoint(200);
    debugger.add_breakpoint(300);

    assert_eq!(debugger.list_breakpoints().len(), 3);
}

#[test]
fn test_debugger_delete_breakpoint() {
    let mut debugger = Debugger::new();
    let id = debugger.add_breakpoint(100);
    assert!(debugger.delete_breakpoint(id));
    assert_eq!(debugger.list_breakpoints().len(), 0);
}

#[test]
fn test_debugger_delete_nonexistent() {
    let mut debugger = Debugger::new();
    assert!(!debugger.delete_breakpoint(999));
}

#[test]
fn test_debugger_enable_disable_breakpoint() {
    let mut debugger = Debugger::new();
    let id = debugger.add_breakpoint(100);

    assert!(debugger.disable_breakpoint(id));
    assert!(!debugger.list_breakpoints()[0].enabled);

    assert!(debugger.enable_breakpoint(id));
    assert!(debugger.list_breakpoints()[0].enabled);
}

#[test]
fn test_debugger_conditional_breakpoint() {
    let mut debugger = Debugger::new();
    let id = debugger.add_conditional_breakpoint(100, "x > 10".to_string());

    let bp = &debugger.list_breakpoints()[0];
    assert!(bp.condition.is_some());
    assert_eq!(bp.condition.as_ref().unwrap(), "x > 10");
}

#[test]
fn test_debugger_watchpoint() {
    let mut debugger = Debugger::new();
    let id = debugger.add_watchpoint("x".to_string());
    assert!(debugger.delete_watchpoint(id));
}

#[test]
fn test_debugger_stepping() {
    let mut debugger = Debugger::new();
    assert!(!debugger.should_break());

    debugger.set_stepping(true);
    assert!(debugger.should_break());
}

#[test]
fn test_debugger_step_over() {
    let mut debugger = Debugger::new();
    debugger.set_step_over();
    assert!(debugger.should_break());
}

#[test]
fn test_debugger_step_out() {
    let mut debugger = Debugger::new();
    debugger.set_step_out();
    assert!(debugger.should_break());
}

#[test]
fn test_debugger_call_depth() {
    let mut debugger = Debugger::new();
    assert_eq!(debugger.call_depth, 0);

    debugger.enter_function();
    assert_eq!(debugger.call_depth, 1);

    debugger.enter_function();
    assert_eq!(debugger.call_depth, 2);

    debugger.exit_function();
    assert_eq!(debugger.call_depth, 1);
}

#[test]
fn test_debugger_format_help() {
    let debugger = Debugger::new();
    let help = debugger.format_help();
    assert!(help.contains("step"));
    assert!(help.contains("break"));
    assert!(help.contains("continue"));
}

#[test]
fn test_debugger_format_breakpoints() {
    let mut debugger = Debugger::new();
    debugger.add_breakpoint(100);
    debugger.add_breakpoint(200);

    let output = debugger.format_breakpoints();
    assert!(output.contains("Breakpoints"));
    assert!(output.contains("#1"));
    assert!(output.contains("#2"));
}

#[test]
fn test_debugger_parse_command_step() {
    let cmd = Debugger::parse_command("s");
    assert!(matches!(cmd, Some(DebugCommand::Step)));

    let cmd = Debugger::parse_command("step");
    assert!(matches!(cmd, Some(DebugCommand::Step)));
}

#[test]
fn test_debugger_parse_command_continue() {
    let cmd = Debugger::parse_command("c");
    assert!(matches!(cmd, Some(DebugCommand::Continue)));

    let cmd = Debugger::parse_command("continue");
    assert!(matches!(cmd, Some(DebugCommand::Continue)));
}

#[test]
fn test_debugger_parse_command_break() {
    let cmd = Debugger::parse_command("b 100");
    assert!(matches!(cmd, Some(DebugCommand::Break(100))));
}

#[test]
fn test_debugger_parse_command_help() {
    let cmd = Debugger::parse_command("h");
    assert!(matches!(cmd, Some(DebugCommand::Help)));
}

#[test]
fn test_debugger_parse_command_quit() {
    let cmd = Debugger::parse_command("q");
    assert!(matches!(cmd, Some(DebugCommand::Quit)));
}

#[test]
fn test_debugger_parse_command_invalid() {
    let cmd = Debugger::parse_command("invalid");
    assert!(cmd.is_none());
}

#[test]
fn test_debugger_parse_command_empty() {
    let cmd = Debugger::parse_command("");
    assert!(cmd.is_none());
}

#[test]
fn test_debugger_session() {
    let mut session = DebuggerSession::new();
    assert!(!session.is_running());

    session.start();
    assert!(session.is_running());

    session.stop();
    assert!(!session.is_running());
}

#[test]
fn test_debugger_session_execute_command() {
    let mut session = DebuggerSession::new();
    session.start();

    session.execute_command(DebugCommand::Break(100));
    assert_eq!(session.debugger().list_breakpoints().len(), 1);
}

#[test]
fn test_debugger_session_step() {
    let mut session = DebuggerSession::new();
    session.start();

    session.execute_command(DebugCommand::Step);
    assert!(session.debugger().should_break());
}

#[test]
fn test_debugger_session_continue() {
    let mut session = DebuggerSession::new();
    session.start();

    session.execute_command(DebugCommand::Step);
    session.execute_command(DebugCommand::Continue);
    assert!(!session.debugger().should_break());
}

#[test]
fn test_debugger_event_history() {
    let mut debugger = Debugger::new();

    debugger.record_event(DebugEvent::BreakpointHit {
        id: 1,
        address: 100,
    });
    debugger.record_event(DebugEvent::StepComplete { address: 101 });

    assert_eq!(debugger.history().len(), 2);
}

#[test]
fn test_debugger_clear_history() {
    let mut debugger = Debugger::new();

    debugger.record_event(DebugEvent::StepComplete { address: 100 });
    debugger.clear_history();

    assert_eq!(debugger.history().len(), 0);
}
