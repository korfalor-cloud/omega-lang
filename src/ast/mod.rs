use crate::errors::Span;
use std::fmt;

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    // Literals
    IntegerLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    CharLiteral(char),
    NoneLiteral,
    BigIntLiteral(String),

    // Identifiers
    Identifier(String),
    SelfExpr,
    SuperExpr,

    // Binary Operations
    BinaryOp {
        op: BinaryOp,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },

    // Unary Operations
    UnaryOp {
        op: UnaryOp,
        operand: Box<AstNode>,
    },

    // Assignment
    Assign {
        target: Box<AstNode>,
        value: Box<AstNode>,
    },

    // Compound Assignment
    CompoundAssign {
        op: BinaryOp,
        target: Box<AstNode>,
        value: Box<AstNode>,
    },

    // Function Call
    Call {
        function: Box<AstNode>,
        args: Vec<AstNode>,
        kwargs: Vec<(String, AstNode)>,
    },

    // Method Call
    MethodCall {
        object: Box<AstNode>,
        method: String,
        args: Vec<AstNode>,
        kwargs: Vec<(String, AstNode)>,
    },

    // Index Access
    Index {
        object: Box<AstNode>,
        index: Box<AstNode>,
    },

    // Slice
    Slice {
        object: Box<AstNode>,
        start: Option<Box<AstNode>>,
        stop: Option<Box<AstNode>>,
        step: Option<Box<AstNode>>,
    },

    // Attribute Access
    Attribute {
        object: Box<AstNode>,
        attribute: String,
    },

    // Optional Chain
    OptionalChain {
        object: Box<AstNode>,
        attribute: String,
    },

    // Ternary
    Ternary {
        condition: Box<AstNode>,
        then_expr: Box<AstNode>,
        else_expr: Box<AstNode>,
    },

    // If Expression
    IfExpr {
        condition: Box<AstNode>,
        then_branch: Box<AstNode>,
        elif_branches: Vec<(AstNode, AstNode)>,
        else_branch: Option<Box<AstNode>>,
    },

    // Match Expression
    MatchExpr {
        scrutinee: Box<AstNode>,
        arms: Vec<MatchArm>,
    },

    // Lambda / Closure
    Lambda {
        params: Vec<Param>,
        body: Box<AstNode>,
        captures: Vec<String>,
    },

    // Async Block
    AsyncBlock {
        body: Box<AstNode>,
    },

    // Await
    Await {
        expr: Box<AstNode>,
    },

    // Yield
    Yield {
        value: Option<Box<AstNode>>,
    },

    // Struct Literal
    StructLiteral {
        name: String,
        fields: Vec<(String, AstNode)>,
        base: Option<Box<AstNode>>,
    },

    // Enum Variant
    EnumVariant {
        enum_name: String,
        variant: String,
        data: Option<Box<AstNode>>,
    },

    // Tuple
    Tuple(Vec<AstNode>),

    // Array / Vec
    Array(Vec<AstNode>),

    // Array Repeat
    ArrayRepeat {
        value: Box<AstNode>,
        count: Box<AstNode>,
    },

    // Map / Dictionary
    Map(Vec<(AstNode, AstNode)>),

    // Set
    Set(Vec<AstNode>),

    // Range
    Range {
        start: Box<AstNode>,
        end: Box<AstNode>,
        inclusive: bool,
    },

    // List Comprehension
    ListComp {
        element: Box<AstNode>,
        iter: Box<AstNode>,
        variable: String,
        condition: Option<Box<AstNode>>,
    },

    // Map Comprehension
    MapComp {
        key: Box<AstNode>,
        value: Box<AstNode>,
        iter: Box<AstNode>,
        variable: String,
        condition: Option<Box<AstNode>>,
    },

    // Generator Expression
    Generator {
        element: Box<AstNode>,
        iter: Box<AstNode>,
        variable: String,
        condition: Option<Box<AstNode>>,
    },

    // Cast / As
    Cast {
        expr: Box<AstNode>,
        target_type: TypeAnnotation,
    },

    // Type Check / Is
    TypeCheck {
        expr: Box<AstNode>,
        check_type: TypeAnnotation,
    },

    // Try Expression
    TryExpr {
        expr: Box<AstNode>,
    },

    // Statements
    LetBinding {
        name: String,
        mutable: bool,
        type_annotation: Option<TypeAnnotation>,
        value: Option<Box<AstNode>>,
    },

    ConstBinding {
        name: String,
        type_annotation: Option<TypeAnnotation>,
        value: Box<AstNode>,
    },

    // Function Definition
    FunctionDef {
        name: String,
        type_params: Vec<TypeParam>,
        params: Vec<Param>,
        return_type: Option<TypeAnnotation>,
        body: Box<AstNode>,
        is_async: bool,
        is_pub: bool,
    },

    // Struct Definition
    StructDef {
        name: String,
        type_params: Vec<TypeParam>,
        fields: Vec<StructField>,
        methods: Vec<AstNode>,
        is_pub: bool,
    },

    // Enum Definition
    EnumDef {
        name: String,
        type_params: Vec<TypeParam>,
        variants: Vec<EnumVariant>,
        is_pub: bool,
    },

    // Trait Definition
    TraitDef {
        name: String,
        type_params: Vec<TypeParam>,
        supertraits: Vec<String>,
        items: Vec<TraitItem>,
        is_pub: bool,
    },

    // Impl Block
    ImplBlock {
        type_params: Vec<TypeParam>,
        trait_name: Option<String>,
        target_type: TypeAnnotation,
        items: Vec<AstNode>,
    },

    // Type Alias
    TypeAlias {
        name: String,
        type_params: Vec<TypeParam>,
        value: TypeAnnotation,
        is_pub: bool,
    },

    // Module
    Module {
        name: String,
        body: Vec<AstNode>,
    },

    // Use / Import
    UseDecl {
        path: Vec<String>,
        alias: Option<String>,
        items: Option<Vec<(String, Option<String>)>>,
    },

    // Block
    Block(Vec<AstNode>),

    // If Statement
    If {
        condition: AstNode,
        then_branch: Box<AstNode>,
        elif_branches: Vec<(AstNode, AstNode)>,
        else_branch: Option<Box<AstNode>>,
    },

    // While Loop
    While {
        condition: AstNode,
        body: Box<AstNode>,
    },

    // For Loop
    For {
        variable: String,
        iterable: AstNode,
        body: Box<AstNode>,
    },

    // Loop (infinite)
    Loop {
        body: Box<AstNode>,
    },

    // Break
    Break {
        value: Option<Box<AstNode>>,
    },

    // Continue
    Continue,

    // Return
    Return {
        value: Option<Box<AstNode>>,
    },

    // Throw
    Throw {
        value: Box<AstNode>,
    },

    // Try/Catch/Finally
    TryCatch {
        try_body: Box<AstNode>,
        catch_clauses: Vec<CatchClause>,
        finally_body: Option<Box<AstNode>>,
    },

    // Defer
    Defer {
        body: Box<AstNode>,
    },

    // Errdefer
    Errdefer {
        body: Box<AstNode>,
    },

    // Assert
    Assert {
        condition: Box<AstNode>,
        message: Option<Box<AstNode>>,
    },

    // Print
    Print {
        args: Vec<AstNode>,
        newline: bool,
    },

    // Test Block
    TestBlock {
        name: String,
        body: Box<AstNode>,
    },

    // Program (top level)
    Program(Vec<AstNode>),

    // Empty
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    FloorDiv,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Spaceship,
    And,
    Or,
    In,
    NotIn,
    Is,
    IsNot,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Pow => write!(f, "**"),
            BinaryOp::FloorDiv => write!(f, "//"),
            BinaryOp::BitAnd => write!(f, "&"),
            BinaryOp::BitOr => write!(f, "|"),
            BinaryOp::BitXor => write!(f, "^"),
            BinaryOp::Shl => write!(f, "<<"),
            BinaryOp::Shr => write!(f, ">>"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::Spaceship => write!(f, "<=>"),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::In => write!(f, "in"),
            BinaryOp::NotIn => write!(f, "not in"),
            BinaryOp::Is => write!(f, "is"),
            BinaryOp::IsNot => write!(f, "is not"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
    Ref,
    Deref,
    Move,
    Copy,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::BitNot => write!(f, "~"),
            UnaryOp::PreInc => write!(f, "++"),
            UnaryOp::PreDec => write!(f, "--"),
            UnaryOp::PostInc => write!(f, "++"),
            UnaryOp::PostDec => write!(f, "--"),
            UnaryOp::Ref => write!(f, "ref"),
            UnaryOp::Deref => write!(f, "deref"),
            UnaryOp::Move => write!(f, "move"),
            UnaryOp::Copy => write!(f, "copy"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub default: Option<Box<AstNode>>,
    pub is_mut: bool,
    pub is_ref: bool,
    pub variadic: bool,
}

impl Param {
    pub fn new(name: String) -> Self {
        Self {
            name,
            type_annotation: None,
            default: None,
            is_mut: false,
            is_ref: false,
            variadic: false,
        }
    }

    pub fn with_type(mut self, ty: TypeAnnotation) -> Self {
        self.type_annotation = Some(ty);
        self
    }

    pub fn with_default(mut self, default: AstNode) -> Self {
        self.default = Some(Box::new(default));
        self
    }

    pub fn mutable(mut self) -> Self {
        self.is_mut = true;
        self
    }

    pub fn by_ref(mut self) -> Self {
        self.is_ref = true;
        self
    }

    pub fn variadic(mut self) -> Self {
        self.variadic = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeAnnotation {
    pub kind: TypeAnnotationKind,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotationKind {
    Simple(String),
    Generic {
        base: Box<TypeAnnotation>,
        args: Vec<TypeAnnotation>,
    },
    Tuple(Vec<TypeAnnotation>),
    Function {
        params: Vec<TypeAnnotation>,
        return_type: Box<TypeAnnotation>,
    },
    Array {
        element: Box<TypeAnnotation>,
        size: Option<Box<AstNode>>,
    },
    Reference {
        mutable: bool,
        inner: Box<TypeAnnotation>,
    },
    Optional(Box<TypeAnnotation>),
    Result {
        ok: Box<TypeAnnotation>,
        err: Box<TypeAnnotation>,
    },
    Union(Vec<TypeAnnotation>),
    Intersection(Vec<TypeAnnotation>),
    SelfType,
    Infer,
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub bounds: Vec<String>,
    pub default: Option<TypeAnnotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub type_annotation: TypeAnnotation,
    pub default: Option<Box<AstNode>>,
    pub is_pub: bool,
    pub is_mut: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub data: Option<EnumVariantData>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnumVariantData {
    Tuple(Vec<TypeAnnotation>),
    Struct(Vec<StructField>),
    Value(Box<AstNode>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraitItem {
    Method {
        name: String,
        params: Vec<Param>,
        return_type: Option<TypeAnnotation>,
        body: Option<Box<AstNode>>,
    },
    AssociatedType {
        name: String,
        bounds: Vec<String>,
        default: Option<TypeAnnotation>,
    },
    Const {
        name: String,
        type_annotation: TypeAnnotation,
        value: Option<Box<AstNode>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<AstNode>>,
    pub body: Box<AstNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Literal(AstNode),
    Identifier(String),
    EnumVariant {
        enum_name: Option<String>,
        variant: String,
        data: Option<Box<Pattern>>,
    },
    Tuple(Vec<Pattern>),
    Array(Vec<Pattern>),
    Struct {
        name: String,
        fields: Vec<(String, Pattern)>,
        rest: bool,
    },
    Or(Vec<Pattern>),
    Range {
        start: Box<AstNode>,
        end: Box<AstNode>,
        inclusive: bool,
    },
    Guard {
        pattern: Box<Pattern>,
        condition: Box<AstNode>,
    },
    Binding {
        name: String,
        pattern: Box<Pattern>,
    },
    Reference {
        mutable: bool,
        pattern: Box<Pattern>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchClause {
    pub binding: Option<String>,
    pub type_annotation: Option<TypeAnnotation>,
    pub body: Box<AstNode>,
}

// Visitor pattern for AST traversal
pub trait AstVisitor<T> {
    fn visit_node(&mut self, node: &AstNode) -> T;

    fn visit_integer(&mut self, value: i64) -> T;
    fn visit_float(&mut self, value: f64) -> T;
    fn visit_string(&mut self, value: &str) -> T;
    fn visit_bool(&mut self, value: bool) -> T;
    fn visit_identifier(&mut self, name: &str) -> T;
    fn visit_binary_op(&mut self, op: BinaryOp, left: &AstNode, right: &AstNode) -> T;
    fn visit_unary_op(&mut self, op: UnaryOp, operand: &AstNode) -> T;
    fn visit_call(&mut self, function: &AstNode, args: &[AstNode]) -> T;
    fn visit_block(&mut self, statements: &[AstNode]) -> T;
    fn visit_if(&mut self, condition: &AstNode, then_branch: &AstNode, else_branch: Option<&AstNode>) -> T;
    fn visit_while(&mut self, condition: &AstNode, body: &AstNode) -> T;
    fn visit_for(&mut self, variable: &str, iterable: &AstNode, body: &AstNode) -> T;
    fn visit_function_def(&mut self, name: &str, params: &[Param], body: &AstNode) -> T;
    fn visit_let_binding(&mut self, name: &str, value: Option<&AstNode>) -> T;
    fn visit_return(&mut self, value: Option<&AstNode>) -> T;
}

// Pretty printer for AST
pub struct AstPrinter {
    indent: usize,
    output: String,
}

impl AstPrinter {
    pub fn new() -> Self {
        Self { indent: 0, output: String::new() }
    }

    pub fn print(&mut self, node: &AstNode) -> String {
        self.print_node(node);
        self.output.clone()
    }

    fn print_node(&mut self, node: &AstNode) {
        let indent = "  ".repeat(self.indent);
        match node {
            AstNode::IntegerLiteral(v) => self.output.push_str(&format!("{}Int({})\n", indent, v)),
            AstNode::FloatLiteral(v) => self.output.push_str(&format!("{}Float({})\n", indent, v)),
            AstNode::StringLiteral(v) => self.output.push_str(&format!("{}String(\"{}\")\n", indent, v)),
            AstNode::BoolLiteral(v) => self.output.push_str(&format!("{}Bool({})\n", indent, v)),
            AstNode::NoneLiteral => self.output.push_str(&format!("{}None\n", indent)),
            AstNode::Identifier(name) => self.output.push_str(&format!("{}Ident({})\n", indent, name)),
            AstNode::BinaryOp { op, left, right } => {
                self.output.push_str(&format!("{}BinaryOp({})\n", indent, op));
                self.indent += 1;
                self.print_node(left);
                self.print_node(right);
                self.indent -= 1;
            }
            AstNode::UnaryOp { op, operand } => {
                self.output.push_str(&format!("{}UnaryOp({})\n", indent, op));
                self.indent += 1;
                self.print_node(operand);
                self.indent -= 1;
            }
            AstNode::Block(stmts) => {
                self.output.push_str(&format!("{}Block({} statements)\n", indent, stmts.len()));
                self.indent += 1;
                for stmt in stmts {
                    self.print_node(stmt);
                }
                self.indent -= 1;
            }
            AstNode::FunctionDef { name, params, body, .. } => {
                self.output.push_str(&format!("{}FnDef({}, {} params)\n", indent, name, params.len()));
                self.indent += 1;
                self.print_node(body);
                self.indent -= 1;
            }
            AstNode::If { condition, then_branch, else_branch, .. } => {
                self.output.push_str(&format!("{}If\n", indent));
                self.indent += 1;
                self.print_node(condition);
                self.print_node(then_branch);
                if let Some(else_b) = else_branch {
                    self.print_node(else_b);
                }
                self.indent -= 1;
            }
            AstNode::While { condition, body } => {
                self.output.push_str(&format!("{}While\n", indent));
                self.indent += 1;
                self.print_node(condition);
                self.print_node(body);
                self.indent -= 1;
            }
            AstNode::For { variable, iterable, body } => {
                self.output.push_str(&format!("{}For({} in ...)\n", indent, variable));
                self.indent += 1;
                self.print_node(iterable);
                self.print_node(body);
                self.indent -= 1;
            }
            AstNode::LetBinding { name, value, .. } => {
                self.output.push_str(&format!("{}Let({})\n", indent, name));
                if let Some(v) = value {
                    self.indent += 1;
                    self.print_node(v);
                    self.indent -= 1;
                }
            }
            AstNode::Return { value } => {
                self.output.push_str(&format!("{}Return\n", indent));
                if let Some(v) = value {
                    self.indent += 1;
                    self.print_node(v);
                    self.indent -= 1;
                }
            }
            AstNode::Call { function, args, .. } => {
                self.output.push_str(&format!("{}Call({} args)\n", indent, args.len()));
                self.indent += 1;
                self.print_node(function);
                for arg in args {
                    self.print_node(arg);
                }
                self.indent -= 1;
            }
            AstNode::Array(elements) => {
                self.output.push_str(&format!("{}Array({} elements)\n", indent, elements.len()));
                self.indent += 1;
                for elem in elements {
                    self.print_node(elem);
                }
                self.indent -= 1;
            }
            AstNode::Map(entries) => {
                self.output.push_str(&format!("{}Map({} entries)\n", indent, entries.len()));
                self.indent += 1;
                for (k, v) in entries {
                    self.print_node(k);
                    self.print_node(v);
                }
                self.indent -= 1;
            }
            AstNode::Program(stmts) => {
                self.output.push_str(&format!("{}Program({} items)\n", indent, stmts.len()));
                self.indent += 1;
                for stmt in stmts {
                    self.print_node(stmt);
                }
                self.indent -= 1;
            }
            _ => {
                self.output.push_str(&format!("{}{:?}\n", indent, node));
            }
        }
    }
}
