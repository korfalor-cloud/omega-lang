use std::io::{self, Write};
use rustyline::error::ReadlineError;
use rustyline::Editor;
use crate::compiler::codegen::CodeGenerator;
use crate::parser::Parser;
use crate::vm::VirtualMachine;
use crate::vm::stack::Value;

pub struct Repl {
    editor: Editor<()>,
    vm: VirtualMachine,
    codegen: CodeGenerator,
    history_file: Option<String>,
    show_ast: bool,
    show_bytecode: bool,
    show_stack: bool,
}

impl Repl {
    pub fn new() -> Self {
        Self {
            editor: Editor::<()>::new(),
            vm: VirtualMachine::new(),
            codegen: CodeGenerator::new(),
            history_file: None,
            show_ast: false,
            show_bytecode: false,
            show_stack: false,
        }
    }

    pub fn with_debug(mut self) -> Self {
        self.vm = VirtualMachine::new().with_debug();
        self.show_ast = true;
        self.show_bytecode = true;
        self.show_stack = true;
        self
    }

    pub fn run(&mut self) {
        self.print_banner();

        loop {
            match self.readline() {
                Ok(input) => {
                    let input = input.trim();
                    if input.is_empty() {
                        continue;
                    }

                    match self.handle_command(input) {
                        Some(result) => {
                            if let Some(output) = result {
                                println!("{}", output);
                            }
                        }
                        None => {
                            self.execute(input);
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }
    }

    fn print_banner(&self) {
        println!("Omega Programming Language v0.1.0");
        println!("Type 'help' for available commands, 'exit' to quit");
        println!();
    }

    fn readline(&mut self) -> Result<String, ReadlineError> {
        let prompt = if self.vm.stack_size() > 0 {
            format!("omega:{}) ", self.vm.stack_size())
        } else {
            "omega> ".to_string()
        };
        self.editor.readline(&prompt)
    }

    fn handle_command(&mut self, input: &str) -> Option<Option<String>> {
        match input {
            "exit" | "quit" => {
                println!("Goodbye!");
                std::process::exit(0);
            }
            "help" => {
                Some(Some(self.help_text()))
            }
            "clear" => {
                print!("\x1B[2J\x1B[1;1H");
                io::stdout().flush().unwrap();
                Some(None)
            }
            "stack" => {
                Some(Some(format!("Stack size: {}", self.vm.stack_size())))
            }
            "frames" => {
                Some(Some(format!("Call frames: {}", self.vm.frame_count())))
            }
            "heap" => {
                Some(Some(format!("Heap objects: {}", self.vm.heap_allocated())))
            }
            "ast" => {
                self.show_ast = !self.show_ast;
                Some(Some(format!("AST display: {}", if self.show_ast { "on" } else { "off" })))
            }
            "bytecode" => {
                self.show_bytecode = !self.show_bytecode;
                Some(Some(format!("Bytecode display: {}", if self.show_bytecode { "on" } else { "off" })))
            }
            "debug" => {
                self.show_ast = true;
                self.show_bytecode = true;
                self.show_stack = true;
                self.vm = VirtualMachine::new().with_debug();
                Some(Some("Debug mode enabled".to_string()))
            }
            "nodebug" => {
                self.show_ast = false;
                self.show_bytecode = false;
                self.show_stack = false;
                self.vm = VirtualMachine::new();
                Some(Some("Debug mode disabled".to_string()))
            }
            _ if input.starts_with("let ") || input.starts_with("fn ") ||
                 input.starts_with("struct ") || input.starts_with("enum ") ||
                 input.starts_with("impl ") || input.starts_with("trait ") ||
                 input.starts_with("mod ") || input.starts_with("use ") ||
                 input.starts_with("test ") => {
                // Multi-line input handling
                None
            }
            _ => None,
        }
    }

    fn execute(&mut self, input: &str) {
        // Parse
        let mut parser = Parser::new(input);
        let ast = match parser.parse() {
            Ok(ast) => ast,
            Err(e) => {
                eprintln!("Parse error: {}", e);
                return;
            }
        };

        if self.show_ast {
            let mut printer = crate::ast::AstPrinter::new();
            println!("AST:\n{}", printer.print(&ast));
        }

        // Compile
        let mut codegen = CodeGenerator::new();
        if let Err(e) = codegen.compile(&ast) {
            eprintln!("Compilation error: {}", e);
            return;
        }

        let chunks = codegen.get_chunks();

        if self.show_bytecode {
            for chunk in chunks {
                println!("{}", chunk.disassemble());
            }
        }

        // Execute
        match self.vm.run(chunks) {
            Ok(value) => {
                match &value {
                    Value::None => {}
                    _ => println!("{}", value.format_display()),
                }

                if self.show_stack {
                    println!("Stack: {}", self.vm.stack_size());
                }
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
            }
        }
    }

    fn help_text(&self) -> String {
        r#"Available commands:
  help     - Show this help
  exit     - Exit the REPL
  clear    - Clear the screen
  stack    - Show stack size
  frames   - Show call frame count
  heap     - Show heap object count
  ast      - Toggle AST display
  bytecode - Toggle bytecode display
  debug    - Enable debug mode
  nodebug  - Disable debug mode

Language features:
  let x = value           - Variable binding
  fn name(args) { body }  - Function definition
  if cond { } else { }    - Conditional
  while cond { }          - While loop
  for x in iter { }       - For loop
  [1, 2, 3]               - Array literal
  {key: value}            - Map literal
  (1, "hello")            - Tuple literal
  print(value)            - Print value
  assert(condition)       - Assert condition"#.to_string()
    }
}
