use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IrNode {
    // Constants
    ConstInteger(i64),
    ConstFloat(f64),
    ConstBool(bool),
    ConstString(String),
    ConstNone,

    // Variables
    LoadLocal(String),
    StoreLocal(String, Box<IrNode>),
    LoadGlobal(String),
    StoreGlobal(String, Box<IrNode>),
    LoadUpvalue(String),
    SetUpvalue(String, Box<IrNode>),

    // Arithmetic
    Add(Box<IrNode>, Box<IrNode>),
    Sub(Box<IrNode>, Box<IrNode>),
    Mul(Box<IrNode>, Box<IrNode>),
    Div(Box<IrNode>, Box<IrNode>),
    Mod(Box<IrNode>, Box<IrNode>),
    Pow(Box<IrNode>, Box<IrNode>),
    Neg(Box<IrNode>),

    // Bitwise
    BitAnd(Box<IrNode>, Box<IrNode>),
    BitOr(Box<IrNode>, Box<IrNode>),
    BitXor(Box<IrNode>, Box<IrNode>),
    BitNot(Box<IrNode>),
    Shl(Box<IrNode>, Box<IrNode>),
    Shr(Box<IrNode>, Box<IrNode>),

    // Comparison
    Eq(Box<IrNode>, Box<IrNode>),
    Ne(Box<IrNode>, Box<IrNode>),
    Lt(Box<IrNode>, Box<IrNode>),
    Le(Box<IrNode>, Box<IrNode>),
    Gt(Box<IrNode>, Box<IrNode>),
    Ge(Box<IrNode>, Box<IrNode>),

    // Logical
    And(Box<IrNode>, Box<IrNode>),
    Or(Box<IrNode>, Box<IrNode>),
    Not(Box<IrNode>),

    // Control Flow
    If {
        condition: Box<IrNode>,
        then_branch: Vec<IrNode>,
        else_branch: Vec<IrNode>,
    },
    While {
        condition: Box<IrNode>,
        body: Vec<IrNode>,
    },
    For {
        variable: String,
        iterable: Box<IrNode>,
        body: Vec<IrNode>,
    },
    Return(Option<Box<IrNode>>),
    Break,
    Continue,

    // Functions
    Call {
        function: Box<IrNode>,
        args: Vec<IrNode>,
    },
    Closure {
        name: String,
        params: Vec<String>,
        body: Vec<IrNode>,
        upvalues: Vec<String>,
    },

    // Data
    Array(Vec<IrNode>),
    Map(Vec<(IrNode, IrNode)>),
    Tuple(Vec<IrNode>),
    Index(Box<IrNode>, Box<IrNode>),
    Field(Box<IrNode>, String),

    // Object
    StructInit {
        name: String,
        fields: Vec<(String, IrNode)>,
    },
    EnumInit {
        name: String,
        variant: String,
        args: Vec<IrNode>,
    },

    // Error Handling
    Try {
        body: Vec<IrNode>,
        catch_clauses: Vec<CatchClause>,
        finally: Option<Vec<IrNode>>,
    },
    Throw(Box<IrNode>),

    // Pattern Matching
    Match {
        value: Box<IrNode>,
        arms: Vec<MatchArm>,
    },

    // Type Operations
    TypeCast(Box<IrNode>, String),
    TypeCheck(Box<IrNode>, String),
    TypeOf(Box<IrNode>),

    // I/O
    Print(Vec<IrNode>, bool),

    // Misc
    Block(Vec<IrNode>),
    Phi(Vec<IrNode>), // SSA phi node
    Nop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchClause {
    pub pattern: Option<String>,
    pub variable: Option<String>,
    pub body: Vec<IrNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<IrNode>,
    pub body: Vec<IrNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Literal(IrNode),
    Identifier(String),
    Wildcard,
    Tuple(Vec<Pattern>),
    Array(Vec<Pattern>),
    Struct {
        name: String,
        fields: Vec<(String, Pattern)>,
    },
    Enum {
        name: String,
        variant: String,
        inner: Vec<Pattern>,
    },
    Range {
        start: Box<IrNode>,
        end: Box<IrNode>,
        inclusive: bool,
    },
    Or(Vec<Pattern>),
    Guard {
        pattern: Box<Pattern>,
        condition: Box<IrNode>,
    },
}

impl IrNode {
    pub fn is_pure(&self) -> bool {
        match self {
            IrNode::ConstInteger(_)
            | IrNode::ConstFloat(_)
            | IrNode::ConstBool(_)
            | IrNode::ConstString(_)
            | IrNode::ConstNone
            | IrNode::LoadLocal(_)
            | IrNode::LoadGlobal(_)
            | IrNode::LoadUpvalue(_) => true,

            IrNode::Add(l, r)
            | IrNode::Sub(l, r)
            | IrNode::Mul(l, r)
            | IrNode::Div(l, r)
            | IrNode::Mod(l, r)
            | IrNode::Pow(l, r)
            | IrNode::BitAnd(l, r)
            | IrNode::BitOr(l, r)
            | IrNode::BitXor(l, r)
            | IrNode::Shl(l, r)
            | IrNode::Shr(l, r)
            | IrNode::Eq(l, r)
            | IrNode::Ne(l, r)
            | IrNode::Lt(l, r)
            | IrNode::Le(l, r)
            | IrNode::Gt(l, r)
            | IrNode::Ge(l, r)
            | IrNode::And(l, r)
            | IrNode::Or(l, r)
            | IrNode::Index(l, r) => l.is_pure() && r.is_pure(),

            IrNode::Neg(n)
            | IrNode::BitNot(n)
            | IrNode::Not(n)
            | IrNode::TypeOf(n)
            | IrNode::Field(n, _) => n.is_pure(),

            _ => false,
        }
    }

    pub fn children(&self) -> Vec<&IrNode> {
        match self {
            IrNode::Add(l, r) | IrNode::Sub(l, r) | IrNode::Mul(l, r) | IrNode::Div(l, r)
            | IrNode::Mod(l, r) | IrNode::Pow(l, r) | IrNode::BitAnd(l, r)
            | IrNode::BitOr(l, r) | IrNode::BitXor(l, r) | IrNode::Shl(l, r)
            | IrNode::Shr(l, r) | IrNode::Eq(l, r) | IrNode::Ne(l, r) | IrNode::Lt(l, r)
            | IrNode::Le(l, r) | IrNode::Gt(l, r) | IrNode::Ge(l, r) | IrNode::And(l, r)
            | IrNode::Or(l, r) | IrNode::Index(l, r) => vec![l.as_ref(), r.as_ref()],

            IrNode::Neg(n) | IrNode::BitNot(n) | IrNode::Not(n) | IrNode::TypeOf(n)
            | IrNode::Throw(n) | IrNode::Field(n, _) => vec![n.as_ref()],

            IrNode::Array(elements) => elements.iter().collect(),
            IrNode::Tuple(elements) => elements.iter().collect(),
            IrNode::Map(entries) => entries.iter().flat_map(|(k, v)| vec![k, v]).collect(),

            IrNode::Call { function, args } => {
                let mut children = vec![function.as_ref()];
                children.extend(args.iter());
                children
            }

            IrNode::Print(args, _) => args.iter().collect(),

            IrNode::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut children = vec![condition.as_ref()];
                children.extend(then_branch.iter());
                children.extend(else_branch.iter());
                children
            }

            _ => vec![],
        }
    }

    pub fn walk<F>(&self, f: &mut F)
    where
        F: FnMut(&IrNode),
    {
        f(self);
        for child in self.children() {
            child.walk(f);
        }
    }
}

impl fmt::Display for IrNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrNode::ConstInteger(n) => write!(f, "{}", n),
            IrNode::ConstFloat(n) => write!(f, "{}", n),
            IrNode::ConstBool(b) => write!(f, "{}", b),
            IrNode::ConstString(s) => write!(f, "\"{}\"", s),
            IrNode::ConstNone => write!(f, "none"),
            IrNode::LoadLocal(name) => write!(f, "local({})", name),
            IrNode::StoreLocal(name, val) => write!(f, "local({}) = {}", name, val),
            IrNode::LoadGlobal(name) => write!(f, "global({})", name),
            IrNode::StoreGlobal(name, val) => write!(f, "global({}) = {}", name, val),
            IrNode::Add(l, r) => write!(f, "({} + {})", l, r),
            IrNode::Sub(l, r) => write!(f, "({} - {})", l, r),
            IrNode::Mul(l, r) => write!(f, "({} * {})", l, r),
            IrNode::Div(l, r) => write!(f, "({} / {})", l, r),
            IrNode::Mod(l, r) => write!(f, "({} % {})", l, r),
            IrNode::Neg(n) => write!(f, "(-{})", n),
            IrNode::Eq(l, r) => write!(f, "({} == {})", l, r),
            IrNode::Ne(l, r) => write!(f, "({} != {})", l, r),
            IrNode::Lt(l, r) => write!(f, "({} < {})", l, r),
            IrNode::Le(l, r) => write!(f, "({} <= {})", l, r),
            IrNode::Gt(l, r) => write!(f, "({} > {})", l, r),
            IrNode::Ge(l, r) => write!(f, "({} >= {})", l, r),
            IrNode::And(l, r) => write!(f, "({} && {})", l, r),
            IrNode::Or(l, r) => write!(f, "({} || {})", l, r),
            IrNode::Not(n) => write!(f, "(!{})", n),
            IrNode::Call { function, args } => {
                write!(f, "{}(", function)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            IrNode::Array(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            IrNode::Return(Some(val)) => write!(f, "return {}", val),
            IrNode::Return(None) => write!(f, "return"),
            IrNode::Block(stmts) => {
                writeln!(f, "{{")?;
                for stmt in stmts {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            _ => write!(f, "{:?}", self),
        }
    }
}
