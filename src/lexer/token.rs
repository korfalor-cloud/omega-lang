use std::fmt;
use strum::{Display, EnumIter, EnumString};

#[derive(Debug, Clone, PartialEq, Display, EnumIter, EnumString)]
pub enum TokenKind {
    // Literals
    #[strum(serialize = "integer")]
    Integer(i64),
    #[strum(serialize = "float")]
    Float(f64),
    #[strum(serialize = "string")]
    String(String),
    #[strum(serialize = "bool")]
    Bool(bool),
    #[strum(serialize = "char")]
    Char(char),
    #[strum(serialize = "byte")]
    Byte(u8),
    #[strum(serialize = "bigint")]
    BigInt(String),

    // Identifier and Keywords
    #[strum(serialize = "identifier")]
    Identifier(String),
    #[strum(serialize = "keyword")]
    Keyword(Keyword),

    // Arithmetic Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,       // **
    SlashSlash,     // //
    Tilde,          // ~

    // Bitwise Operators
    Ampersand,
    Pipe,
    Caret,
    LessLess,       // <<
    GreaterGreater, // >>
    AmpersandAmpersand, // &&

    // Comparison Operators
    Equal,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Spaceship,      // <=>

    // Logical Operators
    Bang,
    AmpAmp,
    PipePipe,

    // Assignment Operators
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    AmpersandEqual,
    PipeEqual,
    CaretEqual,
    LessLessEqual,
    GreaterGreaterEqual,
    StarStarEqual,

    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Semicolon,
    Colon,
    Comma,
    Dot,
    DotDot,
    DotDotDot,
    DotDotEqual,
    Arrow,
    FatArrow,
    Question,
    QuestionDot,
    QuestionQuestion,
    At,
    Hash,
    Dollar,
    Backslash,

    // Special
    Newline,
    Indent,
    Dedent,
    Eof,
    Error(String),
}

impl TokenKind {
    pub fn is_literal(&self) -> bool {
        matches!(self,
            TokenKind::Integer(_) |
            TokenKind::Float(_) |
            TokenKind::String(_) |
            TokenKind::Bool(_) |
            TokenKind::Char(_) |
            TokenKind::Byte(_) |
            TokenKind::BigInt(_)
        )
    }

    pub fn is_operator(&self) -> bool {
        matches!(self,
            TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash |
            TokenKind::Percent | TokenKind::StarStar | TokenKind::SlashSlash |
            TokenKind::Ampersand | TokenKind::Pipe | TokenKind::Caret |
            TokenKind::LessLess | TokenKind::GreaterGreater |
            TokenKind::Equal | TokenKind::EqualEqual | TokenKind::BangEqual |
            TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual |
            TokenKind::Spaceship |
            TokenKind::Bang | TokenKind::AmpAmp | TokenKind::PipePipe |
            TokenKind::PlusEqual | TokenKind::MinusEqual | TokenKind::StarEqual |
            TokenKind::SlashEqual | TokenKind::PercentEqual |
            TokenKind::AmpersandEqual | TokenKind::PipeEqual | TokenKind::CaretEqual |
            TokenKind::LessLessEqual | TokenKind::GreaterGreaterEqual | TokenKind::StarStarEqual
        )
    }

    pub fn is_assignment(&self) -> bool {
        matches!(self,
            TokenKind::Equal | TokenKind::PlusEqual | TokenKind::MinusEqual |
            TokenKind::StarEqual | TokenKind::SlashEqual | TokenKind::PercentEqual |
            TokenKind::AmpersandEqual | TokenKind::PipeEqual | TokenKind::CaretEqual |
            TokenKind::LessLessEqual | TokenKind::GreaterGreaterEqual | TokenKind::StarStarEqual
        )
    }

    pub fn is_comparison(&self) -> bool {
        matches!(self,
            TokenKind::EqualEqual | TokenKind::BangEqual |
            TokenKind::Less | TokenKind::LessEqual |
            TokenKind::Greater | TokenKind::GreaterEqual |
            TokenKind::Spaceship
        )
    }

    pub fn precedence(&self) -> u8 {
        match self {
            TokenKind::PipePipe => 1,
            TokenKind::AmpAmp => 2,
            TokenKind::Pipe => 3,
            TokenKind::Caret => 4,
            TokenKind::Ampersand => 5,
            TokenKind::EqualEqual | TokenKind::BangEqual => 6,
            TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual | TokenKind::Spaceship => 7,
            TokenKind::LessLess | TokenKind::GreaterGreater => 8,
            TokenKind::Plus | TokenKind::Minus => 9,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent | TokenKind::SlashSlash => 10,
            TokenKind::StarStar => 11,
            _ => 0,
        }
    }

    pub fn is_right_associative(&self) -> bool {
        matches!(self, TokenKind::StarStar | TokenKind::Equal)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, EnumIter, EnumString)]
pub enum Keyword {
    // Declarations
    #[strum(serialize = "let")]
    Let,
    #[strum(serialize = "mut")]
    Mut,
    #[strum(serialize = "const")]
    Const,
    #[strum(serialize = "fn")]
    Fn,
    #[strum(serialize = "struct")]
    Struct,
    #[strum(serialize = "enum")]
    Enum,
    #[strum(serialize = "trait")]
    Trait,
    #[strum(serialize = "impl")]
    Impl,
    #[strum(serialize = "type")]
    Type,
    #[strum(serialize = "mod")]
    Mod,
    #[strum(serialize = "use")]
    Use,
    #[strum(serialize = "pub")]
    Pub,
    #[strum(serialize = "static")]
    Static,
    #[strum(serialize = "extern")]
    Extern,

    // Control Flow
    #[strum(serialize = "if")]
    If,
    #[strum(serialize = "else")]
    Else,
    #[strum(serialize = "match")]
    Match,
    #[strum(serialize = "for")]
    For,
    #[strum(serialize = "while")]
    While,
    #[strum(serialize = "loop")]
    Loop,
    #[strum(serialize = "break")]
    Break,
    #[strum(serialize = "continue")]
    Continue,
    #[strum(serialize = "return")]
    Return,
    #[strum(serialize = "yield")]
    Yield,
    #[strum(serialize = "async")]
    Async,
    #[strum(serialize = "await")]
    Await,

    // Literals
    #[strum(serialize = "true")]
    True,
    #[strum(serialize = "false")]
    False,
    #[strum(serialize = "none")]
    None,

    // Error Handling
    #[strum(serialize = "try")]
    Try,
    #[strum(serialize = "catch")]
    Catch,
    #[strum(serialize = "throw")]
    Throw,
    #[strum(serialize = "finally")]
    Finally,

    // Misc
    #[strum(serialize = "as")]
    As,
    #[strum(serialize = "in")]
    In,
    #[strum(serialize = "is")]
    Is,
    #[strum(serialize = "not")]
    Not,
    #[strum(serialize = "and")]
    And,
    #[strum(serialize = "or")]
    Or,
    #[strum(serialize = "where")]
    Where,
    #[strum(serialize = "with")]
    With,
    #[strum(serialize = "class")]
    Class,
    #[strum(serialize = "extends")]
    Extends,
    #[strum(serialize = "new")]
    New,
    #[strum(serialize = "delete")]
    Delete,
    #[strum(serialize = "typeof")]
    Typeof,
    #[strum(serialize = "instanceof")]
    Instanceof,
    #[strum(serialize = "super")]
    Super,
    #[strum(serialize = "self")]
    Self,
    #[strum(serialize = "Self")]
    SelfType,
    #[strum(serialize = "ref")]
    Ref,
    #[strum(serialize = "deref")]
    Deref,
    #[strum(serialize = "move")]
    Move,
    #[strum(serialize = "copy")]
    Copy,
    #[strum(serialize = "clone")]
    Clone,
    #[strum(serialize = "drop")]
    Drop,
    #[strum(serialize = "unsafe")]
    Unsafe,
    #[strum(serialize = "safe")]
    Safe,
    #[strum(serialize = "defer")]
    Defer,
    #[strum(serialize = "errdefer")]
    Errdefer,
    #[strum(serialize = "test")]
    Test,
    #[strum(serialize = "assert")]
    Assert,
    #[strum(serialize = "assert_eq")]
    AssertEq,
    #[strum(serialize = "assert_ne")]
    AssertNe,
    #[strum(serialize = "print")]
    Print,
    #[strum(serialize = "println")]
    Println,
    #[strum(serialize = "eprint")]
    Eprint,
    #[strum(serialize = "eprintln")]
    Eprintln,
    #[strum(serialize = "format")]
    Format,
    #[strum(serialize = "vec")]
    Vec,
    #[strum(serialize = "map")]
    Map,
    #[strum(serialize = "set")]
    Set,
    #[strum(serialize = "result")]
    Result,
    #[strum(serialize = "option")]
    Option,
    #[strum(serialize = "ok")]
    Ok,
    #[strum(serialize = "err")]
    Err,
    #[strum(serialize = "some")]
    Some,
    #[strum(serialize = "global")]
    Global,
    #[strum(serialize = "local")]
    Local,
    #[strum(serialize = "scope")]
    Scope,
    #[strum(serialize = "package")]
    Package,
    #[strum(serialize = "import")]
    Import,
    #[strum(serialize = "from")]
    From,
    #[strum(serialize = "as")]
    Alias,
}

impl Keyword {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "let" => Some(Keyword::Let),
            "mut" => Some(Keyword::Mut),
            "const" => Some(Keyword::Const),
            "fn" => Some(Keyword::Fn),
            "struct" => Some(Keyword::Struct),
            "enum" => Some(Keyword::Enum),
            "trait" => Some(Keyword::Trait),
            "impl" => Some(Keyword::Impl),
            "type" => Some(Keyword::Type),
            "mod" => Some(Keyword::Mod),
            "use" => Some(Keyword::Use),
            "pub" => Some(Keyword::Pub),
            "static" => Some(Keyword::Static),
            "extern" => Some(Keyword::Extern),
            "if" => Some(Keyword::If),
            "else" => Some(Keyword::Else),
            "match" => Some(Keyword::Match),
            "for" => Some(Keyword::For),
            "while" => Some(Keyword::While),
            "loop" => Some(Keyword::Loop),
            "break" => Some(Keyword::Break),
            "continue" => Some(Keyword::Continue),
            "return" => Some(Keyword::Return),
            "yield" => Some(Keyword::Yield),
            "async" => Some(Keyword::Async),
            "await" => Some(Keyword::Await),
            "true" => Some(Keyword::True),
            "false" => Some(Keyword::False),
            "none" => Some(Keyword::None),
            "try" => Some(Keyword::Try),
            "catch" => Some(Keyword::Catch),
            "throw" => Some(Keyword::Throw),
            "finally" => Some(Keyword::Finally),
            "as" => Some(Keyword::As),
            "in" => Some(Keyword::In),
            "is" => Some(Keyword::Is),
            "not" => Some(Keyword::Not),
            "and" => Some(Keyword::And),
            "or" => Some(Keyword::Or),
            "where" => Some(Keyword::Where),
            "with" => Some(Keyword::With),
            "class" => Some(Keyword::Class),
            "extends" => Some(Keyword::Extends),
            "new" => Some(Keyword::New),
            "delete" => Some(Keyword::Delete),
            "typeof" => Some(Keyword::Typeof),
            "instanceof" => Some(Keyword::Instanceof),
            "super" => Some(Keyword::Super),
            "self" => Some(Keyword::Self),
            "Self" => Some(Keyword::SelfType),
            "ref" => Some(Keyword::Ref),
            "deref" => Some(Keyword::Deref),
            "move" => Some(Keyword::Move),
            "copy" => Some(Keyword::Copy),
            "clone" => Some(Keyword::Clone),
            "drop" => Some(Keyword::Drop),
            "unsafe" => Some(Keyword::Unsafe),
            "safe" => Some(Keyword::Safe),
            "defer" => Some(Keyword::Defer),
            "errdefer" => Some(Keyword::Errdefer),
            "test" => Some(Keyword::Test),
            "assert" => Some(Keyword::Assert),
            "assert_eq" => Some(Keyword::AssertEq),
            "assert_ne" => Some(Keyword::AssertNe),
            "print" => Some(Keyword::Print),
            "println" => Some(Keyword::Println),
            "eprint" => Some(Keyword::Eprint),
            "eprintln" => Some(Keyword::Eprintln),
            "format" => Some(Keyword::Format),
            "vec" => Some(Keyword::Vec),
            "map" => Some(Keyword::Map),
            "set" => Some(Keyword::Set),
            "result" => Some(Keyword::Result),
            "option" => Some(Keyword::Option),
            "ok" => Some(Keyword::Ok),
            "err" => Some(Keyword::Err),
            "some" => Some(Keyword::Some),
            "global" => Some(Keyword::Global),
            "local" => Some(Keyword::Local),
            "scope" => Some(Keyword::Scope),
            "package" => Some(Keyword::Package),
            "import" => Some(Keyword::Import),
            "from" => Some(Keyword::From),
            "as" => Some(Keyword::Alias),
            _ => None,
        }
    }

    pub fn is_declaration(&self) -> bool {
        matches!(self,
            Keyword::Let | Keyword::Mut | Keyword::Const | Keyword::Fn |
            Keyword::Struct | Keyword::Enum | Keyword::Trait | Keyword::Impl |
            Keyword::Type | Keyword::Mod | Keyword::Static | Keyword::Class
        )
    }

    pub fn is_control_flow(&self) -> bool {
        matches!(self,
            Keyword::If | Keyword::Else | Keyword::Match | Keyword::For |
            Keyword::While | Keyword::Loop | Keyword::Break | Keyword::Continue |
            Keyword::Return | Keyword::Yield
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub line: usize,
    pub col: usize,
    pub offset: usize,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: String, line: usize, col: usize, offset: usize) -> Self {
        Self { kind, lexeme, line, col, offset }
    }

    pub fn span(&self) -> (usize, usize) {
        (self.offset, self.offset + self.lexeme.len())
    }

    pub fn is_keyword(&self) -> bool {
        matches!(self.kind, TokenKind::Keyword(_))
    }

    pub fn is_literal(&self) -> bool {
        self.kind.is_literal()
    }

    pub fn is_operator(&self) -> bool {
        self.kind.is_operator()
    }

    pub fn keyword(&self) -> Option<Keyword> {
        match &self.kind {
            TokenKind::Keyword(kw) => Some(*kw),
            _ => None,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.lexeme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_from_str() {
        assert_eq!(Keyword::from_str("let"), Some(Keyword::Let));
        assert_eq!(Keyword::from_str("if"), Some(Keyword::If));
        assert_eq!(Keyword::from_str("notakeyword"), None);
    }

    #[test]
    fn test_token_precedence() {
        assert!(TokenKind::StarStar.precedence() > TokenKind::Star.precedence());
        assert!(TokenKind::Star.precedence() > TokenKind::Plus.precedence());
        assert!(TokenKind::Plus.precedence() > TokenKind::Less.precedence());
    }

    #[test]
    fn test_token_properties() {
        assert!(TokenKind::Integer(42).is_literal());
        assert!(!TokenKind::Plus.is_literal());
        assert!(TokenKind::Plus.is_operator());
        assert!(TokenKind::PlusEqual.is_assignment());
        assert!(TokenKind::EqualEqual.is_comparison());
    }
}
