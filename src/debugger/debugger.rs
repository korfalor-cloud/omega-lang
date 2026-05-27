use std::collections::{HashMap, HashSet};
use crate::compiler::bytecode::{Bytecode, Instruction, Constant};
use crate::vm::stack::Value;
use crate::vm::machine::VirtualMachine;
use crate::errors::OmegaResult;

#[derive(Debug, Clone, PartialEq)]
pub enum DebugCommand {
    Step,
    StepOver,
    StepOut,
    Continue,
    Break(usize),
    DeleteBreakpoint(usize),
    ListBreakpoints,
    Print(String),
    Stack,
    Frames,
    Heap,
    Locals,
    Watch(String),
    Unwatch(String),
    Watches,
    Disassemble,
    Memory,
    Help,
    Quit,
}

#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: usize,
    pub address: usize,
    pub chunk_index: usize,
    pub condition: Option<String>,
    pub hit_count: usize,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Watchpoint {
    pub id: usize,
    pub variable: String,
    pub last_value: Option<String>,
}

#[derive(Debug)]
pub struct Debugger {
    breakpoints: HashMap<usize, Breakpoint>,
    watchpoints: HashMap<usize, Watchpoint>,
    next_bp_id: usize,
    next_wp_id: usize,
    stepping: bool,
    step_over: bool,
    step_out: bool,
    call_depth: usize,
    history: Vec<DebugEvent>,
    source_map: HashMap<usize, SourceLocation>,
    output_buffer: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub enum DebugEvent {
    BreakpointHit {
        id: usize,
        address: usize,
    },
    StepComplete {
        address: usize,
    },
    Exception {
        message: String,
    },
    ProgramExit {
        value: Option<String>,
    },
}

impl Debugger {
    pub fn new() -> Self {
        Self {
            breakpoints: HashMap::new(),
            watchpoints: HashMap::new(),
            next_bp_id: 1,
            next_wp_id: 1,
            stepping: false,
            step_over: false,
            step_out: false,
            call_depth: 0,
            history: Vec::new(),
            source_map: HashMap::new(),
            output_buffer: Vec::new(),
        }
    }

    pub fn add_breakpoint(&mut self, address: usize) -> usize {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        self.breakpoints.insert(
            id,
            Breakpoint {
                id,
                address,
                chunk_index: 0,
                condition: None,
                hit_count: 0,
                enabled: true,
            },
        );
        id
    }

    pub fn add_conditional_breakpoint(&mut self, address: usize, condition: String) -> usize {
        let id = self.next_bp_id;
        self.next_bp_id += 1;
        self.breakpoints.insert(
            id,
            Breakpoint {
                id,
                address,
                chunk_index: 0,
                condition: Some(condition),
                hit_count: 0,
                enabled: true,
            },
        );
        id
    }

    pub fn delete_breakpoint(&mut self, id: usize) -> bool {
        self.breakpoints.remove(&id).is_some()
    }

    pub fn enable_breakpoint(&mut self, id: usize) -> bool {
        if let Some(bp) = self.breakpoints.get_mut(&id) {
            bp.enabled = true;
            true
        } else {
            false
        }
    }

    pub fn disable_breakpoint(&mut self, id: usize) -> bool {
        if let Some(bp) = self.breakpoints.get_mut(&id) {
            bp.enabled = false;
            true
        } else {
            false
        }
    }

    pub fn list_breakpoints(&self) -> Vec<&Breakpoint> {
        self.breakpoints.values().collect()
    }

    pub fn add_watchpoint(&mut self, variable: String) -> usize {
        let id = self.next_wp_id;
        self.next_wp_id += 1;
        self.watchpoints.insert(
            id,
            Watchpoint {
                id,
                variable,
                last_value: None,
            },
        );
        id
    }

    pub fn delete_watchpoint(&mut self, id: usize) -> bool {
        self.watchpoints.remove(&id).is_some()
    }

    pub fn check_breakpoint(&mut self, address: usize, chunk_index: usize) -> Option<usize> {
        for bp in self.breakpoints.values_mut() {
            if bp.enabled && bp.address == address && bp.chunk_index == chunk_index {
                bp.hit_count += 1;
                return Some(bp.id);
            }
        }
        None
    }

    pub fn should_break(&self) -> bool {
        self.stepping || self.step_over || self.step_out
    }

    pub fn set_stepping(&mut self, stepping: bool) {
        self.stepping = stepping;
    }

    pub fn set_step_over(&mut self) {
        self.step_over = true;
        self.stepping = false;
    }

    pub fn set_step_out(&mut self) {
        self.step_out = true;
        self.stepping = false;
    }

    pub fn enter_function(&mut self) {
        self.call_depth += 1;
    }

    pub fn exit_function(&mut self) {
        if self.call_depth > 0 {
            self.call_depth -= 1;
        }
        if self.step_out && self.call_depth == 0 {
            self.step_out = false;
            self.stepping = true;
        }
    }

    pub fn record_event(&mut self, event: DebugEvent) {
        self.history.push(event);
    }

    pub fn history(&self) -> &[DebugEvent] {
        &self.history
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn load_source_map(&mut self, map: HashMap<usize, SourceLocation>) {
        self.source_map = map;
    }

    pub fn get_source_location(&self, address: usize) -> Option<&SourceLocation> {
        self.source_map.get(&address)
    }

    pub fn parse_command(input: &str) -> Option<DebugCommand> {
        let parts: Vec<&str> = input.trim().split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        match parts[0] {
            "s" | "step" => Some(DebugCommand::Step),
            "n" | "next" | "step-over" => Some(DebugCommand::StepOver),
            "out" | "step-out" => Some(DebugCommand::StepOut),
            "c" | "continue" => Some(DebugCommand::Continue),
            "b" | "break" => {
                if parts.len() > 1 {
                    parts[1].parse().ok().map(DebugCommand::Break)
                } else {
                    None
                }
            }
            "d" | "delete" => {
                if parts.len() > 1 {
                    parts[1].parse().ok().map(DebugCommand::DeleteBreakpoint)
                } else {
                    None
                }
            }
            "info" | "i" => {
                if parts.len() > 1 {
                    match parts[1] {
                        "b" | "break" | "breakpoints" => Some(DebugCommand::ListBreakpoints),
                        "s" | "stack" => Some(DebugCommand::Stack),
                        "f" | "frames" => Some(DebugCommand::Frames),
                        "l" | "locals" => Some(DebugCommand::Locals),
                        "w" | "watch" => Some(DebugCommand::Watches),
                        "m" | "mem" | "memory" => Some(DebugCommand::Memory),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            "p" | "print" => {
                if parts.len() > 1 {
                    Some(DebugCommand::Print(parts[1..].join(" ")))
                } else {
                    None
                }
            }
            "w" | "watch" => {
                if parts.len() > 1 {
                    Some(DebugCommand::Watch(parts[1].to_string()))
                } else {
                    None
                }
            }
            "unwatch" => {
                if parts.len() > 1 {
                    Some(DebugCommand::Unwatch(parts[1].to_string()))
                } else {
                    None
                }
            }
            "dis" | "disassemble" => Some(DebugCommand::Disassemble),
            "h" | "help" => Some(DebugCommand::Help),
            "q" | "quit" => Some(DebugCommand::Quit),
            _ => None,
        }
    }

    pub fn format_help(&self) -> String {
        r#"Debugger Commands:
  s, step          - Step one instruction
  n, next          - Step over function calls
  out, step-out    - Step out of current function
  c, continue      - Continue execution
  b, break <addr>  - Set breakpoint at address
  d, delete <id>   - Delete breakpoint
  info b           - List breakpoints
  info s           - Show stack
  info f           - Show call frames
  info l           - Show local variables
  info w           - Show watchpoints
  info m           - Show memory usage
  p, print <expr>  - Print expression
  w, watch <var>   - Watch variable
  unwatch <id>     - Remove watchpoint
  dis              - Disassemble current chunk
  h, help          - Show this help
  q, quit          - Exit debugger"#
            .to_string()
    }

    pub fn format_stack(&self, stack: &[Value]) -> String {
        let mut output = String::from("Stack:\n");
        for (i, value) in stack.iter().enumerate() {
            output.push_str(&format!("  [{}] {:?}\n", i, value));
        }
        output
    }

    pub fn format_breakpoints(&self) -> String {
        let mut output = String::from("Breakpoints:\n");
        for bp in self.breakpoints.values() {
            let status = if bp.enabled { "enabled" } else { "disabled" };
            let cond = bp
                .condition
                .as_ref()
                .map(|c| format!(" if {}", c))
                .unwrap_or_default();
            output.push_str(&format!(
                "  #{} at {} [{}] (hits: {}){}\n",
                bp.id, bp.address, status, bp.hit_count, cond
            ));
        }
        output
    }

    pub fn format_watchpoints(&self) -> String {
        let mut output = String::from("Watchpoints:\n");
        for wp in self.watchpoints.values() {
            let last = wp.last_value.as_deref().unwrap_or("N/A");
            output.push_str(&format!("  #{} {} = {}\n", wp.id, wp.variable, last));
        }
        output
    }

    pub fn output_buffer(&self) -> &[String] {
        &self.output_buffer
    }

    pub fn clear_output(&mut self) {
        self.output_buffer.clear();
    }

    pub fn push_output(&mut self, line: String) {
        self.output_buffer.push(line);
    }
}

pub struct DebuggerSession {
    debugger: Debugger,
    running: bool,
    current_chunk: usize,
    current_ip: usize,
}

impl DebuggerSession {
    pub fn new() -> Self {
        Self {
            debugger: Debugger::new(),
            running: false,
            current_chunk: 0,
            current_ip: 0,
        }
    }

    pub fn start(&mut self) {
        self.running = true;
        self.debugger.push_output("Debugger started. Type 'help' for commands.".to_string());
    }

    pub fn stop(&mut self) {
        self.running = false;
        self.debugger.push_output("Debugger stopped.".to_string());
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn execute_command(&mut self, command: DebugCommand) {
        match command {
            DebugCommand::Step => {
                self.debugger.set_stepping(true);
                self.debugger.push_output("Stepping...".to_string());
            }
            DebugCommand::StepOver => {
                self.debugger.set_step_over();
                self.debugger.push_output("Stepping over...".to_string());
            }
            DebugCommand::StepOut => {
                self.debugger.set_step_out();
                self.debugger.push_output("Stepping out...".to_string());
            }
            DebugCommand::Continue => {
                self.debugger.set_stepping(false);
                self.debugger.push_output("Continuing...".to_string());
            }
            DebugCommand::Break(addr) => {
                let id = self.debugger.add_breakpoint(addr);
                self.debugger.push_output(format!("Breakpoint #{} set at {}", id, addr));
            }
            DebugCommand::DeleteBreakpoint(id) => {
                if self.debugger.delete_breakpoint(id) {
                    self.debugger.push_output(format!("Breakpoint #{} deleted", id));
                } else {
                    self.debugger.push_output(format!("Breakpoint #{} not found", id));
                }
            }
            DebugCommand::ListBreakpoints => {
                let output = self.debugger.format_breakpoints();
                self.debugger.push_output(output);
            }
            DebugCommand::Stack => {
                self.debugger.push_output("Stack info (requires VM context)".to_string());
            }
            DebugCommand::Frames => {
                self.debugger.push_output("Frame info (requires VM context)".to_string());
            }
            DebugCommand::Locals => {
                self.debugger.push_output("Local variables (requires VM context)".to_string());
            }
            DebugCommand::Watches => {
                let output = self.debugger.format_watchpoints();
                self.debugger.push_output(output);
            }
            DebugCommand::Watch(var) => {
                let id = self.debugger.add_watchpoint(var.clone());
                self.debugger.push_output(format!("Watchpoint #{} set on '{}'", id, var));
            }
            DebugCommand::Unwatch(id) => {
                if self.debugger.delete_watchpoint(id) {
                    self.debugger.push_output(format!("Watchpoint #{} deleted", id));
                } else {
                    self.debugger.push_output(format!("Watchpoint #{} not found", id));
                }
            }
            DebugCommand::Print(expr) => {
                self.debugger.push_output(format!("Print: {} (requires evaluation)", expr));
            }
            DebugCommand::Disassemble => {
                self.debugger.push_output("Disassembly (requires bytecode)".to_string());
            }
            DebugCommand::Memory => {
                self.debugger.push_output("Memory info (requires VM context)".to_string());
            }
            DebugCommand::Help => {
                let help = self.debugger.format_help();
                self.debugger.push_output(help);
            }
            DebugCommand::Quit => {
                self.stop();
            }
        }
    }

    pub fn debugger(&self) -> &Debugger {
        &self.debugger
    }

    pub fn debugger_mut(&mut self) -> &mut Debugger {
        &mut self.debugger
    }
}
