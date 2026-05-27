use crate::ast::*;
use crate::errors::OmegaResult;

pub struct Formatter {
    indent_size: usize,
    current_indent: usize,
    output: String,
    line_width: usize,
}

impl Formatter {
    pub fn new() -> Self {
        Self {
            indent_size: 4,
            current_indent: 0,
            output: String::new(),
            line_width: 80,
        }
    }

    pub fn with_indent_size(mut self, size: usize) -> Self {
        self.indent_size = size;
        self
    }

    pub fn with_line_width(mut self, width: usize) -> Self {
        self.line_width = width;
        self
    }

    pub fn format(&mut self, ast: &AstNode) -> OmegaResult<String> {
        self.output.clear();
        self.current_indent = 0;
        self.visit_node(ast)?;
        Ok(self.output.clone())
    }

    fn visit_node(&mut self, node: &AstNode) -> OmegaResult<()> {
        match node {
            AstNode::Program(stmts) => {
                for (i, stmt) in stmts.iter().enumerate() {
                    if i > 0 {
                        self.newline();
                    }
                    self.visit_node(stmt)?;
                }
            }
            AstNode::Block(stmts) => {
                self.write("{");
                self.indent();
                for stmt in stmts {
                    self.newline();
                    self.visit_node(stmt)?;
                }
                self.dedent();
                self.newline();
                self.write("}");
            }
            AstNode::LetBinding { name, mutable, type_annotation, value } => {
                self.write("let ");
                if *mutable {
                    self.write("mut ");
                }
                self.write(name);
                if let Some(ty) = type_annotation {
                    self.write(": ");
                    self.visit_type(ty)?;
                }
                if let Some(val) = value {
                    self.write(" = ");
                    self.visit_node(val)?;
                }
            }
            AstNode::ConstBinding { name, type_annotation, value } => {
                self.write("const ");
                self.write(name);
                if let Some(ty) = type_annotation {
                    self.write(": ");
                    self.visit_type(ty)?;
                }
                self.write(" = ");
                self.visit_node(value)?;
            }
            AstNode::FunctionDef { name, params, return_type, body, is_async, is_pub } => {
                if *is_pub {
                    self.write("pub ");
                }
                if *is_async {
                    self.write("async ");
                }
                self.write("fn ");
                self.write(name);
                self.write("(");
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    if param.is_mut {
                        self.write("mut ");
                    }
                    self.write(&param.name);
                    if let Some(ty) = &param.type_annotation {
                        self.write(": ");
                        self.visit_type(ty)?;
                    }
                    if let Some(default) = &param.default {
                        self.write(" = ");
                        self.visit_node(default)?;
                    }
                }
                self.write(")");
                if let Some(ret) = return_type {
                    self.write(" -> ");
                    self.visit_type(ret)?;
                }
                self.write(" ");
                self.visit_node(body)?;
            }
            AstNode::If { condition, then_branch, elif_branches, else_branch } => {
                self.write("if ");
                self.visit_node(condition)?;
                self.write(" ");
                self.visit_node(then_branch)?;
                for (elif_cond, elif_body) in elif_branches {
                    self.write(" else if ");
                    self.visit_node(elif_cond)?;
                    self.write(" ");
                    self.visit_node(elif_body)?;
                }
                if let Some(else_b) = else_branch {
                    self.write(" else ");
                    self.visit_node(else_b)?;
                }
            }
            AstNode::While { condition, body } => {
                self.write("while ");
                self.visit_node(condition)?;
                self.write(" ");
                self.visit_node(body)?;
            }
            AstNode::For { variable, iterable, body } => {
                self.write("for ");
                self.write(variable);
                self.write(" in ");
                self.visit_node(iterable)?;
                self.write(" ");
                self.visit_node(body)?;
            }
            AstNode::Loop { body } => {
                self.write("loop ");
                self.visit_node(body)?;
            }
            AstNode::Return { value } => {
                self.write("return");
                if let Some(v) = value {
                    self.write(" ");
                    self.visit_node(v)?;
                }
            }
            AstNode::Break { value } => {
                self.write("break");
                if let Some(v) = value {
                    self.write(" ");
                    self.visit_node(v)?;
                }
            }
            AstNode::Continue => {
                self.write("continue");
            }
            AstNode::Assign { target, value } => {
                self.visit_node(target)?;
                self.write(" = ");
                self.visit_node(value)?;
            }
            AstNode::BinaryOp { op, left, right } => {
                self.visit_node(left)?;
                self.write(&format!(" {} ", op));
                self.visit_node(right)?;
            }
            AstNode::UnaryOp { op, operand } => {
                self.write(&format!("{}", op));
                self.visit_node(operand)?;
            }
            AstNode::Call { function, args, .. } => {
                self.visit_node(function)?;
                self.write("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_node(arg)?;
                }
                self.write(")");
            }
            AstNode::IntegerLiteral(v) => self.write(&v.to_string()),
            AstNode::FloatLiteral(v) => self.write(&v.to_string()),
            AstNode::StringLiteral(v) => self.write(&format!("\"{}\"", v)),
            AstNode::BoolLiteral(v) => self.write(&v.to_string()),
            AstNode::NoneLiteral => self.write("none"),
            AstNode::Identifier(name) => self.write(name),
            AstNode::Array(elements) => {
                self.write("[");
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_node(elem)?;
                }
                self.write("]");
            }
            AstNode::Map(entries) => {
                self.write("{");
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_node(key)?;
                    self.write(": ");
                    self.visit_node(value)?;
                }
                self.write("}");
            }
            AstNode::Tuple(elements) => {
                self.write("(");
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_node(elem)?;
                }
                self.write(")");
            }
            AstNode::Print { args, newline } => {
                if *newline {
                    self.write("println(");
                } else {
                    self.write("print(");
                }
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_node(arg)?;
                }
                self.write(")");
            }
            _ => {
                self.write(&format!("{:?}", node));
            }
        }
        Ok(())
    }

    fn visit_type(&mut self, ty: &TypeAnnotation) -> OmegaResult<()> {
        match &ty.kind {
            TypeAnnotationKind::Simple(name) => self.write(name),
            TypeAnnotationKind::Generic { base, args } => {
                self.visit_type(base)?;
                self.write("<");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_type(arg)?;
                }
                self.write(">");
            }
            TypeAnnotationKind::Optional(inner) => {
                self.visit_type(inner)?;
                self.write("?");
            }
            TypeAnnotationKind::Array { element, .. } => {
                self.write("[");
                self.visit_type(element)?;
                self.write("]");
            }
            TypeAnnotationKind::Tuple(types) => {
                self.write("(");
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.visit_type(t)?;
                }
                self.write(")");
            }
            _ => self.write("Any"),
        }
        Ok(())
    }

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn newline(&mut self) {
        self.output.push('\n');
        for _ in 0..self.current_indent * self.indent_size {
            self.output.push(' ');
        }
    }

    fn indent(&mut self) {
        self.current_indent += 1;
    }

    fn dedent(&mut self) {
        self.current_indent = self.current_indent.saturating_sub(1);
    }
}
