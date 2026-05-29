use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// NFA / DFA Regex Engine  (Thompson's construction)
// ---------------------------------------------------------------------------

/// Unique state identifier inside an NFA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StateId(usize);

/// Labels for NFA transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Label {
    Char(char),
    Any,
    Epsilon,
}

/// A single NFA state with outgoing transitions.
#[derive(Debug, Clone)]
struct NfaState {
    transitions: Vec<(Label, StateId)>,
}

/// Non-deterministic finite automaton produced by Thompson's construction.
#[derive(Debug)]
struct Nfa {
    states: Vec<NfaState>,
    start: StateId,
    accept: StateId,
}

impl Nfa {
    fn new_state(states: &mut Vec<NfaState>) -> StateId {
        let id = StateId(states.len());
        states.push(NfaState {
            transitions: Vec::new(),
        });
        id
    }

    /// Build an NFA that accepts a single character.
    fn from_char(c: char) -> Self {
        let mut states = Vec::new();
        let start = Self::new_state(&mut states);
        let accept = Self::new_state(&mut states);
        states[start.0].transitions.push((Label::Char(c), accept));
        Nfa { states, start, accept }
    }

    /// Build an NFA that matches any single character.
    fn from_any() -> Self {
        let mut states = Vec::new();
        let start = Self::new_state(&mut states);
        let accept = Self::new_state(&mut states);
        states[start.0].transitions.push((Label::Any, accept));
        Nfa { states, start, accept }
    }

    /// Concatenation: first followed by second.
    fn concat(first: Self, second: Self) -> Self {
        let mut states = first.states;
        let offset = StateId(states.len());
        for mut s in second.states {
            for t in &mut s.transitions {
                t.1 = StateId(t.1 .0 + offset.0);
            }
            states.push(s);
        }
        let start = first.start;
        let accept = StateId(second.accept.0 + offset.0);
        // epsilon transition from first.accept -> second.start
        let second_start = StateId(second.start.0 + offset.0);
        states[first.accept.0]
            .transitions
            .push((Label::Epsilon, second_start));
        Nfa { states, start, accept }
    }

    /// Alternation (union) of two NFAs.
    fn union(a: Self, b: Self) -> Self {
        let mut states = Vec::new();
        let new_start = Self::new_state(&mut states);

        // helper to merge an NFA with an offset
        let mut merge = |nfa: Nfa, states: &mut Vec<NfaState>| -> (StateId, StateId) {
            let offset = StateId(states.len());
            for mut s in nfa.states {
                for t in &mut s.transitions {
                    t.1 = StateId(t.1 .0 + offset.0);
                }
                states.push(s);
            }
            (
                StateId(nfa.start.0 + offset.0),
                StateId(nfa.accept.0 + offset.0),
            )
        };

        let (a_start, a_accept) = merge(a, &mut states);
        let (b_start, b_accept) = merge(b, &mut states);
        let new_accept = Self::new_state(&mut states);

        states[new_start.0]
            .transitions
            .push((Label::Epsilon, a_start));
        states[new_start.0]
            .transitions
            .push((Label::Epsilon, b_start));
        states[a_accept.0]
            .transitions
            .push((Label::Epsilon, new_accept));
        states[b_accept.0]
            .transitions
            .push((Label::Epsilon, new_accept));

        Nfa {
            states,
            start: new_start,
            accept: new_accept,
        }
    }

    /// Kleene star (zero or more).
    fn star(nfa: Self) -> Self {
        let mut states = nfa.states;
        let new_start = Self::new_state(&mut states);
        let new_accept = Self::new_state(&mut states);

        states[new_start.0]
            .transitions
            .push((Label::Epsilon, nfa.start));
        states[new_start.0]
            .transitions
            .push((Label::Epsilon, new_accept));
        states[nfa.accept.0]
            .transitions
            .push((Label::Epsilon, nfa.start));
        states[nfa.accept.0]
            .transitions
            .push((Label::Epsilon, new_accept));

        Nfa {
            states,
            start: new_start,
            accept: new_accept,
        }
    }

    /// One-or-more repetition.
    fn plus(nfa: Self) -> Self {
        // a+ = a a*
        let star_nfa = Self::star(nfa.clone());
        Self::concat(nfa, star_nfa)
    }

    /// Zero-or-one (optional).
    fn optional(nfa: Self) -> Self {
        // a? = a | epsilon
        let mut states = nfa.states;
        let new_start = Self::new_state(&mut states);
        let new_accept = Self::new_state(&mut states);

        states[new_start.0]
            .transitions
            .push((Label::Epsilon, nfa.start));
        states[new_start.0]
            .transitions
            .push((Label::Epsilon, new_accept));
        states[nfa.accept.0]
            .transitions
            .push((Label::Epsilon, new_accept));

        Nfa {
            states,
            start: new_start,
            accept: new_accept,
        }
    }

    /// Epsilon-closure: set of states reachable via epsilon transitions.
    fn epsilon_closure(&self, states: &HashSet<StateId>) -> HashSet<StateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<StateId> = states.iter().copied().collect();
        while let Some(sid) = queue.pop_front() {
            for (label, target) in &self.states[sid.0].transitions {
                if *label == Label::Epsilon && !closure.contains(target) {
                    closure.insert(*target);
                    queue.push_back(*target);
                }
            }
        }
        closure
    }
}

impl Clone for Nfa {
    fn clone(&self) -> Self {
        Nfa {
            states: self.states.clone(),
            start: self.start,
            accept: self.accept,
        }
    }
}

// ---------------------------------------------------------------------------
// DFA
// ---------------------------------------------------------------------------

/// DFA state key is a sorted set of NFA state ids.
type DfaKey = Vec<StateId>;

fn dfa_key(set: &HashSet<StateId>) -> DfaKey {
    let mut v: Vec<StateId> = set.iter().copied().collect();
    v.sort_by_key(|s| s.0);
    v
}

struct Dfa {
    transitions: HashMap<(usize, char), usize>,
    accept: HashSet<usize>,
    start: usize,
}

impl Dfa {
    /// Subset construction from an NFA.
    fn from_nfa(nfa: &Nfa) -> Self {
        let mut state_map: HashMap<DfaKey, usize> = HashMap::new();
        let mut transitions: HashMap<(usize, char), usize> = HashMap::new();
        let mut accept: HashSet<usize> = HashSet::new();

        let start_set = nfa.epsilon_closure(&HashSet::from([nfa.start]));
        let start_key = dfa_key(&start_set);
        state_map.insert(start_key.clone(), 0);
        let mut next_id: usize = 1;

        let mut worklist: VecDeque<(DfaKey, HashSet<StateId>)> = VecDeque::new();
        worklist.push_back((start_key, start_set));

        while let Some((key, nfa_set)) = worklist.pop_front() {
            let dfa_state = *state_map.get(&key).unwrap();

            // collect input symbols from transitions
            let mut symbols: HashSet<char> = HashSet::new();
            for sid in &nfa_set {
                for (label, _) in &nfa.states[sid.0] .transitions {
                    match label {
                        Label::Char(c) => { symbols.insert(*c); }
                        Label::Any => { /* handle below */ }
                        _ => {}
                    }
                }
            }

            // check accept
            if nfa_set.contains(&nfa.accept) {
                accept.insert(dfa_state);
            }

            // for a "Char" transition
            for c in symbols {
                let mut next_set = HashSet::new();
                for sid in &nfa_set {
                    for (label, target) in &nfa.states[sid.0].transitions {
                        match label {
                            Label::Char(ch) if *ch == c => { next_set.insert(*target); }
                            _ => {}
                        }
                    }
                }
                let next_closure = nfa.epsilon_closure(&next_set);
                let next_key = dfa_key(&next_closure);
                let is_new = !state_map.contains_key(&next_key);
                if is_new {
                    state_map.insert(next_key.clone(), next_id);
                    next_id += 1;
                }
                transitions.insert((dfa_state, c), *state_map.get(&next_key).unwrap());
                if is_new && !worklist.iter().any(|(k, _)| *k == next_key) {
                    worklist.push_back((next_key.clone(), next_closure));
                }
            }
        }

        Dfa {
            transitions,
            accept,
            start: 0,
        }
    }

    /// Run the DFA on input text, returning whether it accepts.
    fn run(&self, text: &str) -> bool {
        let mut state = self.start;
        for c in text.chars() {
            if let Some(&next) = self.transitions.get(&(state, c)) {
                state = next;
            } else {
                return false;
            }
        }
        self.accept.contains(&state)
    }
}

// ---------------------------------------------------------------------------
// Parser  (recursive-descent: regex -> NFA, with capture groups)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum AstNode {
    Literal(char),
    Dot,
    Concat(Vec<AstNode>),
    Alternation(Box<AstNode>, Box<AstNode>),
    Star(Box<AstNode>),
    Plus(Box<AstNode>),
    Optional(Box<AstNode>),
    Group(usize, Box<AstNode>),
    CharClass(Vec<char>, bool),
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    group_counter: usize,
}

impl Parser {
    fn new(pattern: &str) -> Self {
        Parser {
            chars: pattern.chars().collect(),
            pos: 0,
            group_counter: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        self.pos += 1;
        c
    }

    fn parse(&mut self) -> Result<AstNode, String> {
        let node = self.parse_alternation()?;
        Ok(node)
    }

    fn parse_alternation(&mut self) -> Result<AstNode, String> {
        let mut left = self.parse_concat()?;
        while self.peek() == Some('|') {
            self.advance();
            let right = self.parse_concat()?;
            left = AstNode::Alternation(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_concat(&mut self) -> Result<AstNode, String> {
        let mut parts = Vec::new();
        while let Some(c) = self.peek() {
            if c == ')' || c == '|' {
                break;
            }
            parts.push(self.parse_quantified()?);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(AstNode::Concat(parts))
        }
    }

    fn parse_quantified(&mut self) -> Result<AstNode, String> {
        let mut node = self.parse_atom()?;
        while let Some(c) = self.peek() {
            match c {
                '*' => {
                    self.advance();
                    node = AstNode::Star(Box::new(node));
                }
                '+' => {
                    self.advance();
                    node = AstNode::Plus(Box::new(node));
                }
                '?' => {
                    self.advance();
                    node = AstNode::Optional(Box::new(node));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_atom(&mut self) -> Result<AstNode, String> {
        match self.peek() {
            Some('(') => {
                self.advance();
                self.group_counter += 1;
                let group_num = self.group_counter;
                let inner = self.parse_alternation()?;
                if self.advance() != Some(')') {
                    return Err("Unmatched '('".into());
                }
                Ok(AstNode::Group(group_num, Box::new(inner)))
            }
            Some('[') => {
                self.advance();
                let negated = if self.peek() == Some('^') {
                    self.advance();
                    true
                } else {
                    false
                };
                let mut chars = Vec::new();
                loop {
                    match self.advance() {
                        Some(']') => break,
                        Some('\\') => {
                            if let Some(esc) = self.advance() {
                                chars.push(esc);
                            }
                        }
                        Some(c) => chars.push(c),
                        None => return Err("Unterminated character class".into()),
                    }
                }
                Ok(AstNode::CharClass(chars, negated))
            }
            Some('.') => {
                self.advance();
                Ok(AstNode::Dot)
            }
            Some('\\') => {
                self.advance();
                match self.advance() {
                    Some('d') => {
                        Ok(AstNode::CharClass(('0'..='9').collect(), false))
                    }
                    Some('D') => {
                        Ok(AstNode::CharClass(('0'..='9').collect(), true))
                    }
                    Some('w') => {
                        let mut chars: Vec<char> = ('a'..='z').collect();
                        chars.extend('A'..='Z');
                        chars.extend('0'..='9');
                        chars.push('_');
                        Ok(AstNode::CharClass(chars, false))
                    }
                    Some('W') => {
                        let mut chars: Vec<char> = ('a'..='z').collect();
                        chars.extend('A'..='Z');
                        chars.extend('0'..='9');
                        chars.push('_');
                        Ok(AstNode::CharClass(chars, true))
                    }
                    Some('s') => {
                        Ok(AstNode::CharClass(vec![' ', '\t', '\n', '\r'], false))
                    }
                    Some('S') => {
                        Ok(AstNode::CharClass(vec![' ', '\t', '\n', '\r'], true))
                    }
                    Some('n') => Ok(AstNode::Literal('\n')),
                    Some('t') => Ok(AstNode::Literal('\t')),
                    Some('r') => Ok(AstNode::Literal('\r')),
                    Some(c) => Ok(AstNode::Literal(c)),
                    None => Err("Trailing backslash".into()),
                }
            }
            Some(')') => Err("Unmatched ')'".into()),
            Some(c) => {
                self.advance();
                Ok(AstNode::Literal(c))
            }
            None => Err("Unexpected end of pattern".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation helpers
// ---------------------------------------------------------------------------

fn compile_ast(node: &AstNode) -> Nfa {
    match node {
        AstNode::Literal(c) => Nfa::from_char(*c),
        AstNode::Dot => Nfa::from_any(),
        AstNode::Concat(nodes) => {
            let mut nfa = compile_ast(&nodes[0]);
            for child in &nodes[1..] {
                nfa = Nfa::concat(nfa, compile_ast(child));
            }
            nfa
        }
        AstNode::Alternation(l, r) => Nfa::union(compile_ast(l), compile_ast(r)),
        AstNode::Star(inner) => Nfa::star(compile_ast(inner)),
        AstNode::Plus(inner) => Nfa::plus(compile_ast(inner)),
        AstNode::Optional(inner) => Nfa::optional(compile_ast(inner)),
        AstNode::Group(_, inner) => compile_ast(inner),
        AstNode::CharClass(chars, negated) => {
            // build union of all char literals (or the inverse)
            let all: Vec<char> = ('!'..='~').collect(); // printable ASCII
            let targets = if *negated {
                all.into_iter().filter(|c| !chars.contains(c)).collect::<Vec<_>>()
            } else {
                chars.clone()
            };
            let mut nfa = Nfa::from_char(targets[0]);
            for &c in &targets[1..] {
                nfa = Nfa::union(nfa, Nfa::from_char(c));
            }
            nfa
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// An advanced regular expression engine built with Thompson's NFA construction
/// and optional DFA (subset) construction.
pub struct AdvancedRegex {
    pattern: String,
    nfa: Nfa,
    group_count: usize,
}

impl fmt::Debug for AdvancedRegex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdvancedRegex")
            .field("pattern", &self.pattern)
            .field("group_count", &self.group_count)
            .finish()
    }
}

impl AdvancedRegex {
    /// Compile a regex pattern.
    pub fn new(pattern: &str) -> Result<Self, String> {
        let mut parser = Parser::new(pattern);
        let ast = parser.parse()?;
        let nfa = compile_ast(&ast);
        Ok(AdvancedRegex {
            pattern: pattern.to_string(),
            nfa,
            group_count: parser.group_counter,
        })
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Test whether the pattern matches anywhere in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        self.search(text).is_some()
    }

    /// Find the first match, returning (start, end) byte offsets.
    pub fn search(&self, text: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        for start in 0..=n {
            if let Some(end) = self.try_match_from(&chars, start) {
                return Some((start, end));
            }
        }
        None
    }

    /// Return all non-overlapping matches as (start, end) char-index pairs.
    pub fn find_all(&self, text: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut results = Vec::new();
        let mut pos = 0;
        while pos <= n {
            if let Some(end) = self.try_match_from(&chars, pos) {
                if end == pos {
                    // zero-length match: advance by one to avoid infinite loop
                    results.push((pos, end));
                    pos += 1;
                } else {
                    results.push((pos, end));
                    pos = end;
                }
            } else {
                pos += 1;
            }
        }
        results
    }

    /// Extract capture groups for the first match.
    /// Returns `None` if no match.
    /// The first element is the full match; subsequent elements are group 1, 2, ...
    pub fn captures(&self, text: &str) -> Option<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        for start in 0..=n {
            if let Some((groups, end)) = self.try_match_with_groups(&chars, start) {
                let full: String = chars[start..end].iter().collect();
                let mut result = vec![full];
                let mut sorted_keys: Vec<usize> = groups.keys().copied().collect();
                sorted_keys.sort();
                for k in sorted_keys {
                    result.push(groups[&k].clone());
                }
                return Some(result);
            }
        }
        None
    }

    /// Replace first occurrence.
    pub fn replace(&self, text: &str, replacement: &str) -> String {
        if let Some((start, end)) = self.search(text) {
            let chars: Vec<char> = text.chars().collect();
            let before: String = chars[..start].iter().collect();
            let after: String = chars[end..].iter().collect();
            format!("{}{}{}", before, replacement, after)
        } else {
            text.to_string()
        }
    }

    /// Replace all occurrences.
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let matches = self.find_all(text);
        if matches.is_empty() {
            return text.to_string();
        }
        let mut result = String::new();
        let mut last = 0;
        for (s, e) in matches {
            let seg: String = chars[last..s].iter().collect();
            result.push_str(&seg);
            result.push_str(replacement);
            last = e;
        }
        let tail: String = chars[last..].iter().collect();
        result.push_str(&tail);
        result
    }

    /// Split text by the pattern.
    pub fn split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let matches = self.find_all(text);
        if matches.is_empty() {
            return vec![text.to_string()];
        }
        let mut result = Vec::new();
        let mut last = 0;
        for (s, e) in matches {
            let seg: String = chars[last..s].iter().collect();
            result.push(seg);
            last = e;
        }
        let tail: String = chars[last..].iter().collect();
        result.push(tail);
        result
    }

    // -- internal matching via NFA simulation (with backtracking for groups) --

    fn try_match_from(&self, chars: &[char], start: usize) -> Option<usize> {
        let mut current: HashSet<StateId> = HashSet::new();
        current.insert(self.nfa.start);
        current = self.nfa.epsilon_closure(&current);

        // If the NFA is already in an accept state (zero-length match), return now.
        if current.contains(&self.nfa.accept) {
            return Some(start);
        }

        for i in start..chars.len() {
            let mut next = HashSet::new();
            for sid in &current {
                for (label, target) in &self.nfa.states[sid.0].transitions {
                    match label {
                        Label::Char(c) if *c == chars[i] => {
                            next.insert(*target);
                        }
                        Label::Any => {
                            next.insert(*target);
                        }
                        _ => {}
                    }
                }
            }
            if next.is_empty() {
                return None;
            }
            current = self.nfa.epsilon_closure(&next);
            if current.contains(&self.nfa.accept) {
                return Some(i + 1);
            }
        }
        None
    }

    /// Like `try_match_from` but also collects group captures.
    /// This uses a recursive backtracking approach over the AST to track groups.
    fn try_match_with_groups(&self, chars: &[char], start: usize) -> Option<(Vec<String>, usize)> {
        let mut parser = Parser::new(&self.pattern);
        let ast = parser.parse().ok()?;
        match_ast(&ast, chars, start).map(|(groups, end)| {
            // groups[0] is unused placeholder; real groups start at 1
            let max_group = parser.group_counter;
            let mut result = Vec::new();
            for i in 1..=max_group {
                result.push(groups.get(&i).cloned().unwrap_or_default());
            }
            (result, end)
        })
    }
}

// ---------------------------------------------------------------------------
// AST-based matcher with group tracking  (backtracking)
// ---------------------------------------------------------------------------

type GroupMap = HashMap<usize, String>;

fn match_ast(node: &AstNode, chars: &[char], pos: usize) -> Option<(GroupMap, usize)> {
    match node {
        AstNode::Literal(c) => {
            if pos < chars.len() && chars[pos] == *c {
                Some((GroupMap::new(), pos + 1))
            } else {
                None
            }
        }
        AstNode::Dot => {
            if pos < chars.len() {
                Some((GroupMap::new(), pos + 1))
            } else {
                None
            }
        }
        AstNode::CharClass(class_chars, negated) => {
            if pos < chars.len() {
                let in_class = class_chars.contains(&chars[pos]);
                if in_class != *negated {
                    Some((GroupMap::new(), pos + 1))
                } else {
                    None
                }
            } else {
                None
            }
        }
        AstNode::Concat(nodes) => {
            let mut groups = GroupMap::new();
            let mut p = pos;
            for child in nodes {
                let (g, next) = match_ast(child, chars, p)?;
                groups.extend(g);
                p = next;
            }
            Some((groups, p))
        }
        AstNode::Alternation(left, right) => {
            match_ast(left, chars, pos).or_else(|| match_ast(right, chars, pos))
        }
        AstNode::Star(inner) => match_star(inner, chars, pos),
        AstNode::Plus(inner) => {
            // must match at least once
            let (g1, p1) = match_ast(inner, chars, pos)?;
            let (mut g_rest, p_rest) = match_star(inner, chars, p1);
            g_rest.extend(g1);
            Some((g_rest, p_rest))
        }
        AstNode::Optional(inner) => match_ast(inner, chars, pos)
            .or_else(|| Some((GroupMap::new(), pos))),
        AstNode::Group(num, inner) => {
            let (mut groups, end) = match_ast(inner, chars, pos)?;
            let captured: String = chars[pos..end].iter().collect();
            groups.insert(*num, captured);
            Some((groups, end))
        }
    }
}

fn match_star(inner: &AstNode, chars: &[char], pos: usize) -> Option<(GroupMap, usize)> {
    // greedy: try to match as many repetitions as possible, then backtrack
    let mut best_groups = GroupMap::new();
    let mut best_pos = pos;
    let mut current_groups = GroupMap::new();
    let mut p = pos;

    loop {
        if let Some((g, next)) = match_ast(inner, chars, p) {
            if next == p {
                // zero-length match, stop to avoid infinite loop
                break;
            }
            current_groups.extend(g);
            best_groups = current_groups.clone();
            best_pos = next;
            p = next;
        } else {
            break;
        }
    }
    Some((best_groups, best_pos))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        let re = AdvancedRegex::new("abc").unwrap();
        assert!(re.is_match("abc"));
        assert!(re.is_match("xyzabc"));
        assert!(!re.is_match("ab"));
        assert!(!re.is_match("abd"));
    }

    #[test]
    fn test_dot_wildcard() {
        let re = AdvancedRegex::new("a.c").unwrap();
        assert!(re.is_match("abc"));
        assert!(re.is_match("aXc"));
        assert!(!re.is_match("ac"));
        assert!(!re.is_match("abbc"));
    }

    #[test]
    fn test_star_quantifier() {
        let re = AdvancedRegex::new("ab*c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
    }

    #[test]
    fn test_plus_quantifier() {
        let re = AdvancedRegex::new("ab+c").unwrap();
        assert!(!re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
    }

    #[test]
    fn test_optional_quantifier() {
        let re = AdvancedRegex::new("colou?r").unwrap();
        assert!(re.is_match("color"));
        assert!(re.is_match("colour"));
    }

    #[test]
    fn test_alternation() {
        let re = AdvancedRegex::new("cat|dog").unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("dog"));
        assert!(!re.is_match("bird"));
    }

    #[test]
    fn test_group_capture() {
        let re = AdvancedRegex::new("(\\d+)-(\\d+)").unwrap();
        let caps = re.captures("date: 2024-01").unwrap();
        assert_eq!(caps[0], "2024-01");
        assert_eq!(caps[1], "2024");
        assert_eq!(caps[2], "01");
    }

    #[test]
    fn test_nested_groups() {
        let re = AdvancedRegex::new("((a)(b))").unwrap();
        let caps = re.captures("ab").unwrap();
        assert_eq!(caps[0], "ab");
        assert_eq!(caps[1], "ab");
        assert_eq!(caps[2], "a");
        assert_eq!(caps[3], "b");
    }

    #[test]
    fn test_char_class() {
        let re = AdvancedRegex::new("[abc]+").unwrap();
        assert!(re.is_match("aabbc"));
        assert!(!re.is_match("def"));
    }

    #[test]
    fn test_negated_char_class() {
        let re = AdvancedRegex::new("[^abc]+").unwrap();
        assert!(re.is_match("xyz"));
        assert!(!re.is_match("abc"));
    }

    #[test]
    fn test_search_finds_position() {
        let re = AdvancedRegex::new("world").unwrap();
        assert_eq!(re.search("hello world"), Some((6, 11)));
    }

    #[test]
    fn test_find_all() {
        let re = AdvancedRegex::new("\\d+").unwrap();
        let matches = re.find_all("a12b345c6");
        assert_eq!(matches, vec![(1, 3), (4, 7), (8, 9)]);
    }

    #[test]
    fn test_replace_first() {
        let re = AdvancedRegex::new("world").unwrap();
        assert_eq!(re.replace("hello world!", "rust"), "hello rust!");
    }

    #[test]
    fn test_replace_all() {
        let re = AdvancedRegex::new("\\d+").unwrap();
        assert_eq!(re.replace_all("a1b2c3", "#"), "a#b#c#");
    }

    #[test]
    fn test_split() {
        let re = AdvancedRegex::new(",").unwrap();
        let parts = re.split("a,b,c");
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_empty_pattern() {
        let re = AdvancedRegex::new("").unwrap();
        assert!(re.is_match(""));
        assert!(re.is_match("anything"));
    }

    #[test]
    fn test_group_count() {
        let re = AdvancedRegex::new("(a)(b)(c)").unwrap();
        assert_eq!(re.group_count(), 3);
    }

    #[test]
    fn test_complex_pattern() {
        // match an email-like pattern
        let re = AdvancedRegex::new("[a-z]+@[a-z]+\\.com").unwrap();
        assert!(re.is_match("user@example.com"));
        assert!(!re.is_match("user@"));
    }

    #[test]
    fn test_alternation_in_group() {
        let re = AdvancedRegex::new("(go|went)").unwrap();
        let caps = re.captures("went home").unwrap();
        assert_eq!(caps[0], "went");
        assert_eq!(caps[1], "went");
    }

}
