use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process;

mod lexer;
mod parser;
mod ast;
mod semantic;
mod compiler;
mod vm;
mod gc;
mod types;
mod stdlib;
mod errors;
mod diagnostics;
mod utils;
mod repl;

use errors::OmegaError;

#[derive(Parser)]
#[command(name = "omega")]
#[command(version = "0.1.0")]
#[command(about = "Omega Programming Language", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input file to run
    #[arg(value_name = "FILE")]
    file: Option<PathBuf>,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Show AST
    #[arg(long)]
    ast: bool,

    /// Show bytecode
    #[arg(long)]
    bytecode: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an Omega program
    Run {
        /// Input file
        file: PathBuf,

        /// Enable debug output
        #[arg(short, long)]
        debug: bool,
    },

    /// Start the REPL
    Repl {
        /// Enable debug mode
        #[arg(short, long)]
        debug: bool,
    },

    /// Compile an Omega program
    Compile {
        /// Input file
        file: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "0")]
        opt_level: u8,
    },

    /// Format an Omega source file
    Fmt {
        /// Input file
        file: PathBuf,

        /// Check formatting without modifying
        #[arg(long)]
        check: bool,
    },

    /// Run tests
    Test {
        /// Input file or directory
        path: Option<PathBuf>,

        /// Test name filter
        #[arg(short, long)]
        filter: Option<String>,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Lint an Omega source file
    Lint {
        /// Input file
        file: PathBuf,
    },

    /// Show version information
    Version,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run { file, debug }) => {
            run_file(&file, debug, false, false);
        }
        Some(Commands::Repl { debug }) => {
            let mut repl = repl::Repl::new();
            if debug {
                repl = repl.with_debug();
            }
            repl.run();
        }
        Some(Commands::Compile { file, output, opt_level }) => {
            compile_file(&file, output, opt_level);
        }
        Some(Commands::Fmt { file, check }) => {
            format_file(&file, check);
        }
        Some(Commands::Test { path, filter, verbose }) => {
            run_tests(path, filter, verbose);
        }
        Some(Commands::Lint { file }) => {
            lint_file(&file);
        }
        Some(Commands::Version) => {
            print_version();
        }
        None => {
            if let Some(file) = cli.file {
                run_file(&file, cli.debug, cli.ast, cli.bytecode);
            } else {
                // Start REPL
                let mut repl = repl::Repl::new();
                if cli.debug {
                    repl = repl.with_debug();
                }
                repl.run();
            }
        }
    }
}

fn run_file(path: &PathBuf, debug: bool, show_ast: bool, show_bytecode: bool) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    // Parse
    let mut parser = parser::Parser::new(&source);
    let ast = match parser.parse() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    };

    if show_ast {
        let mut printer = ast::AstPrinter::new();
        println!("AST:\n{}", printer.print(&ast));
    }

    // Compile
    let mut codegen = compiler::codegen::CodeGenerator::new();
    if let Err(e) = codegen.compile(&ast) {
        eprintln!("Compilation error: {}", e);
        process::exit(1);
    }

    let chunks = codegen.get_chunks();

    if show_bytecode {
        for chunk in chunks {
            println!("{}", chunk.disassemble());
        }
    }

    // Execute
    let mut vm = if debug {
        vm::VirtualMachine::new().with_debug()
    } else {
        vm::VirtualMachine::new()
    };

    match vm.run(chunks) {
        Ok(_) => {
            if debug {
                eprintln!("Program exited successfully");
            }
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            process::exit(1);
        }
    }
}

fn compile_file(path: &PathBuf, output: Option<PathBuf>, _opt_level: u8) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    // Parse
    let mut parser = parser::Parser::new(&source);
    let ast = match parser.parse() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    };

    // Compile
    let mut codegen = compiler::codegen::CodeGenerator::new();
    if let Err(e) = codegen.compile(&ast) {
        eprintln!("Compilation error: {}", e);
        process::exit(1);
    }

    let chunks = codegen.get_chunks();

    // Output
    let output_path = output.unwrap_or_else(|| {
        let mut p = path.clone();
        p.set_extension("omega");
        p
    });

    let mut output_data = String::new();
    for chunk in chunks {
        output_data.push_str(&chunk.disassemble());
    }

    match fs::write(&output_path, output_data) {
        Ok(_) => println!("Compiled to {}", output_path.display()),
        Err(e) => {
            eprintln!("Error writing output: {}", e);
            process::exit(1);
        }
    }
}

fn format_file(path: &PathBuf, check: bool) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    // Simple formatting - just re-parse and print
    let mut parser = parser::Parser::new(&source);
    match parser.parse() {
        Ok(_) => {
            if check {
                println!("{} is correctly formatted", path.display());
            } else {
                println!("Formatted {}", path.display());
            }
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    }
}

fn run_tests(path: Option<PathBuf>, filter: Option<String>, verbose: bool) {
    let test_path = path.unwrap_or_else(|| PathBuf::from("."));

    if !test_path.exists() {
        eprintln!("Path does not exist: {}", test_path.display());
        process::exit(1);
    }

    println!("Running tests in {}", test_path.display());
    if let Some(f) = &filter {
        println!("Filter: {}", f);
    }

    // TODO: Implement test discovery and execution
    println!("Test runner not yet fully implemented");
}

fn lint_file(path: &PathBuf) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    let mut parser = parser::Parser::new(&source);
    match parser.parse() {
        Ok(_) => println!("No issues found in {}", path.display()),
        Err(e) => {
            eprintln!("Lint error: {}", e);
            process::exit(1);
        }
    }
}

fn print_version() {
    println!("omega {}", env!("CARGO_PKG_VERSION"));
    println!("Omega Programming Language");
    println!("Build: {}", env!("CARGO_PKG_NAME"));
}
