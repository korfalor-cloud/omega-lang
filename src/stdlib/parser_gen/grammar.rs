/// Parser generator: grammar definition, LL(1) parser, and token generation.

use std::collections::{HashMap, HashSet, BTreeMap};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    Terminal(String),
    NonTerminal(String),
    Epsilon,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Production {
    pub lhs: String,
    pub rhs: Vec<Symbol>,
}

impl Production {
    pub fn new(lhs: &str, rhs: Vec<Symbol>) -> Self {
        Self { lhs: lhs.to_string(), rhs }
    }
}

#[derive(Debug, Clone)]
pub struct Grammar {
    pub productions: Vec<Production>,
    pub start_symbol: String,
    pub terminals: HashSet<String>,
    pub non_terminals: HashSet<String>,
}

impl Grammar {
    pub fn new(start: &str) -> Self {
        Self {
            productions: Vec::new(),
            start_symbol: start.to_string(),
            terminals: HashSet::new(),
            non_terminals: HashSet::new(),
        }
    }

    pub fn add_rule(&mut self, lhs: &str, rhs: Vec<Symbol>) {
        self.non_terminals.insert(lhs.to_string());
        for sym in &rhs {
            match sym {
                Symbol::Terminal(t) => { self.terminals.insert(t.clone()); }
                Symbol::NonTerminal(nt) => { self.non_terminals.insert(nt.clone()); }
                _ => {}
            }
        }
        self.productions.push(Production::new(lhs, rhs));
    }

    pub fn add_rules(&mut self, lhs: &str, alternatives: Vec<Vec<Symbol>>) {
        for alt in alternatives {
            self.add_rule(lhs, alt);
        }
    }

    /// Compute FIRST set for a symbol.
    pub fn first(&self, symbol: &Symbol) -> HashSet<String> {
        let mut result = HashSet::new();
        self.first_of(symbol, &mut result, &mut HashSet::new());
        result
    }

    fn first_of(&self, symbol: &Symbol, result: &mut HashSet<String>, visited: &mut HashSet<String>) {
        match symbol {
            Symbol::Terminal(t) => {
                result.insert(t.clone());
            }
            Symbol::Epsilon => {
                result.insert(String::new()); // epsilon
            }
            Symbol::Eof => {
                result.insert("$".to_string());
            }
            Symbol::NonTerminal(nt) => {
                if !visited.insert(nt.clone()) {
                    return;
                }
                for prod in &self.productions {
                    if prod.lhs == *nt {
                        if prod.rhs.is_empty() || prod.rhs[0] == Symbol::Epsilon {
                            result.insert(String::new());
                        } else {
                            for sym in &prod.rhs {
                                self.first_of(sym, result, visited);
                                if !result.contains("") {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute FIRST set for a sequence of symbols.
    pub fn first_of_sequence(&self, symbols: &[Symbol]) -> HashSet<String> {
        let mut result = HashSet::new();
        if symbols.is_empty() {
            result.insert(String::new());
            return result;
        }
        for sym in symbols {
            let first = self.first(sym);
            let has_epsilon = first.contains("");
            for f in first {
                if !f.is_empty() {
                    result.insert(f);
                }
            }
            if !has_epsilon {
                break;
            }
            if sym == symbols.last().unwrap() {
                result.insert(String::new());
            }
        }
        result
    }

    /// Compute FOLLOW set for a non-terminal.
    pub fn follow(&self, symbol: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        if symbol == self.start_symbol {
            result.insert("$".to_string());
        }
        self.follow_helper(symbol, &mut result, &mut HashSet::new());
        result
    }

    fn follow_helper(&self, symbol: &str, result: &mut HashSet<String>, visited: &mut HashSet<String>) {
        if !visited.insert(symbol.to_string()) {
            return;
        }

        for prod in &self.productions {
            for i in 0..prod.rhs.len() {
                if let Symbol::NonTerminal(nt) = &prod.rhs[i] {
                    if nt == symbol {
                        // Get symbols after this non-terminal
                        let after = &prod.rhs[i + 1..];
                        if after.is_empty() {
                            if prod.lhs != *symbol {
                                let follow_lhs = self.follow(&prod.lhs);
                                result.extend(follow_lhs);
                            }
                        } else {
                            let first_after = self.first_of_sequence(after);
                            let has_epsilon = first_after.contains("");
                            for f in first_after {
                                if !f.is_empty() {
                                    result.insert(f);
                                }
                            }
                            if has_epsilon && prod.lhs != *symbol {
                                let follow_lhs = self.follow(&prod.lhs);
                                result.extend(follow_lhs);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if the grammar is LL(1).
    pub fn is_ll1(&self) -> bool {
        let nt_list: Vec<String> = self.non_terminals.iter().cloned().collect();
        for nt in &nt_list {
            let prods: Vec<&Production> = self.productions.iter().filter(|p| p.lhs == *nt).collect();
            for i in 0..prods.len() {
                for j in (i + 1)..prods.len() {
                    let first_i = self.first_of_sequence(&prods[i].rhs);
                    let first_j = self.first_of_sequence(&prods[j].rhs);

                    // Check if FIRST sets overlap
                    let intersection: HashSet<_> = first_i.intersection(&first_j).collect();
                    if !intersection.is_empty() && !intersection.contains(&&String::new()) {
                        return false;
                    }

                    // If one can derive epsilon, check FOLLOW
                    if first_i.contains("") {
                        let follow_nt = self.follow(nt);
                        let intersection: HashSet<_> = first_j.intersection(&follow_nt).collect();
                        if !intersection.is_empty() {
                            return false;
                        }
                    }
                    if first_j.contains("") {
                        let follow_nt = self.follow(nt);
                        let intersection: HashSet<_> = first_i.intersection(&follow_nt).collect();
                        if !intersection.is_empty() {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Generate LL(1) parse table.
    pub fn parse_table(&self) -> Option<HashMap<(String, String), usize>> {
        let mut table: HashMap<(String, String), usize> = HashMap::new();

        for (idx, prod) in self.productions.iter().enumerate() {
            let first = self.first_of_sequence(&prod.rhs);
            let follow_nt = self.follow(&prod.lhs);

            for terminal in &first {
                if terminal.is_empty() {
                    // Epsilon production: add for all in FOLLOW
                    for f in &follow_nt {
                        let key = (prod.lhs.clone(), f.clone());
                        if table.contains_key(&key) {
                            return None; // Not LL(1)
                        }
                        table.insert(key, idx);
                    }
                } else {
                    let key = (prod.lhs.clone(), terminal.clone());
                    if table.contains_key(&key) {
                        return None; // Not LL(1)
                    }
                    table.insert(key, idx);
                }
            }
        }

        Some(table)
    }
}

/// Token definition for lexer generation.
#[derive(Debug, Clone)]
pub struct TokenDef {
    pub name: String,
    pub pattern: String,
    pub priority: u32,
}

impl TokenDef {
    pub fn new(name: &str, pattern: &str, priority: u32) -> Self {
        Self { name: name.to_string(), pattern: pattern.to_string(), priority }
    }
}

/// Lexer generator that creates token rules.
#[derive(Debug)]
pub struct LexerDef {
    tokens: Vec<TokenDef>,
    skip_patterns: Vec<String>,
}

impl LexerDef {
    pub fn new() -> Self {
        Self { tokens: Vec::new(), skip_patterns: Vec::new() }
    }

    pub fn add_token(&mut self, name: &str, pattern: &str, priority: u32) {
        self.tokens.push(TokenDef::new(name, pattern, priority));
    }

    pub fn add_skip(&mut self, pattern: &str) {
        self.skip_patterns.push(pattern.to_string());
    }

    pub fn tokens(&self) -> &[TokenDef] {
        &self.tokens
    }

    pub fn skip_patterns(&self) -> &[String] {
        &self.skip_patterns
    }

    /// Sort tokens by priority (higher = matched first).
    pub fn sorted_tokens(&self) -> Vec<&TokenDef> {
        let mut sorted: Vec<&TokenDef> = self.tokens.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }
}

impl Default for LexerDef {
    fn default() -> Self {
        Self::new()
    }
}

/// AST node for parse tree.
#[derive(Debug, Clone)]
pub enum AstNode {
    Terminal { value: String, token: String },
    NonTerminal { name: String, children: Vec<AstNode> },
}

impl AstNode {
    pub fn terminal(value: &str, token: &str) -> Self {
        AstNode::Terminal { value: value.to_string(), token: token.to_string() }
    }

    pub fn non_terminal(name: &str, children: Vec<AstNode>) -> Self {
        AstNode::NonTerminal { name: name.to_string(), children }
    }

    pub fn to_string_indented(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            AstNode::Terminal { value, token } => {
                format!("{}{}(\"{}\")", pad, token, value)
            }
            AstNode::NonTerminal { name, children } => {
                let mut s = format!("{}{}\n", pad, name);
                for child in children {
                    s.push_str(&child.to_string_indented(indent + 1));
                    s.push('\n');
                }
                s
            }
        }
    }
}

/// Simple recursive descent parser framework.
pub struct Parser {
    grammar: Grammar,
    tokens: Vec<(String, String)>, // (token_type, value)
    position: usize,
}

impl Parser {
    pub fn new(grammar: Grammar, tokens: Vec<(String, String)>) -> Self {
        Self { grammar, tokens, position: 0 }
    }

    pub fn current_token(&self) -> Option<&(String, String)> {
        self.tokens.get(self.position)
    }

    pub fn advance(&mut self) -> Option<(String, String)> {
        if self.position < self.tokens.len() {
            let tok = self.tokens[self.position].clone();
            self.position += 1;
            Some(tok)
        } else {
            None
        }
    }

    pub fn expect(&mut self, token_type: &str) -> Result<String, String> {
        match self.current_token() {
            Some((t, v)) if t == token_type => {
                let value = v.clone();
                self.advance();
                Ok(value)
            }
            Some((t, _)) => Err(format!("Expected {}, got {}", token_type, t)),
            None => Err(format!("Expected {}, got EOF", token_type)),
        }
    }

    pub fn at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

/// Generate a grammar from a BNF-like description.
pub fn parse_bnf(description: &str) -> Grammar {
    let lines: Vec<&str> = description.lines().collect();
    if lines.is_empty() {
        return Grammar::new("start");
    }

    // Find start symbol from first rule
    let start = lines[0].split_whitespace().next().unwrap_or("start");
    let mut grammar = Grammar::new(start);

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        // Parse: LHS ::= ALT1 | ALT2 | ...
        let parts: Vec<&str> = line.splitn(2, "::=").collect();
        if parts.len() != 2 {
            continue;
        }
        let lhs = parts[0].trim();
        let alternatives = parts[1];

        for alt in alternatives.split('|') {
            let symbols: Vec<Symbol> = alt.split_whitespace().map(|s| {
                let s = s.trim();
                if s.starts_with('\'') || s.starts_with('"') {
                    // Terminal
                    let inner = s.trim_matches(|c| c == '\'' || c == '"');
                    Symbol::Terminal(inner.to_string())
                } else if s == "ε" || s == "epsilon" || s == "EPSILON" {
                    Symbol::Epsilon
                } else if s == "$" || s == "EOF" {
                    Symbol::Eof
                } else {
                    Symbol::NonTerminal(s.to_string())
                }
            }).filter(|s| *s != Symbol::Epsilon || alt.trim() == "ε" || alt.trim() == "epsilon")
            .collect();

            grammar.add_rule(lhs, symbols);
        }
    }

    grammar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_grammar() {
        let mut g = Grammar::new("E");
        g.add_rule("E", vec![Symbol::NonTerminal("T".into()), Symbol::Terminal("+".into()), Symbol::NonTerminal("E".into())]);
        g.add_rule("E", vec![Symbol::NonTerminal("T".into())]);
        g.add_rule("T", vec![Symbol::Terminal("id".into())]);

        let first_e = g.first(&Symbol::NonTerminal("E".into()));
        assert!(first_e.contains("id"));
    }

    #[test]
    fn test_first_set() {
        let mut g = Grammar::new("S");
        g.add_rule("S", vec![Symbol::NonTerminal("A".into()), Symbol::Terminal("b".into())]);
        g.add_rule("A", vec![Symbol::Terminal("a".into())]);
        g.add_rule("A", vec![Symbol::Epsilon]);

        let first_a = g.first(&Symbol::NonTerminal("A".into()));
        assert!(first_a.contains("a"));
        assert!(first_a.contains("")); // epsilon
    }

    #[test]
    fn test_follow_set() {
        let mut g = Grammar::new("S");
        g.add_rule("S", vec![Symbol::NonTerminal("A".into()), Symbol::Terminal("b".into())]);
        g.add_rule("A", vec![Symbol::Terminal("a".into())]);

        let follow_a = g.follow("A");
        assert!(follow_a.contains("b"));
    }

    #[test]
    fn test_parse_table() {
        let mut g = Grammar::new("S");
        g.add_rule("S", vec![Symbol::Terminal("a".into()), Symbol::NonTerminal("S".into())]);
        g.add_rule("S", vec![Symbol::Epsilon]);

        let table = g.parse_table();
        assert!(table.is_some());
    }

    #[test]
    fn test_bnf_parser() {
        let bnf = r#"
            expr ::= term '+' expr | term
            term ::= factor '*' term | factor
            factor ::= '(' expr ')' | 'id'
        "#;
        let g = parse_bnf(bnf);
        assert!(!g.productions.is_empty());
        assert!(g.non_terminals.contains("expr"));
    }

    #[test]
    fn test_lexer_def() {
        let mut lexer = LexerDef::new();
        lexer.add_token("NUMBER", "[0-9]+", 10);
        lexer.add_token("IDENT", "[a-zA-Z_][a-zA-Z0-9_]*", 5);
        lexer.add_token("PLUS", "\\+", 10);
        lexer.add_skip("[ \t\n]+");

        assert_eq!(lexer.tokens().len(), 3);
    }
}
