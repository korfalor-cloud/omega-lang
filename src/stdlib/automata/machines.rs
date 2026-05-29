/// Automata theory: DFA, NFA, pushdown automata, Turing machines.

use std::collections::{HashMap, HashSet, VecDeque};

/// Deterministic Finite Automaton.
#[derive(Debug, Clone)]
pub struct DFA {
    pub states: usize,
    pub alphabet: Vec<char>,
    pub transitions: Vec<HashMap<char, usize>>,
    pub start: usize,
    pub accept: HashSet<usize>,
}

impl DFA {
    pub fn new(states: usize, alphabet: Vec<char>, start: usize, accept: HashSet<usize>) -> Self {
        Self {
            states,
            alphabet,
            transitions: vec![HashMap::new(); states],
            start,
            accept,
        }
    }

    pub fn add_transition(&mut self, from: usize, symbol: char, to: usize) {
        self.transitions[from].insert(symbol, to);
    }

    pub fn accepts(&self, input: &str) -> bool {
        let mut current = self.start;
        for ch in input.chars() {
            if let Some(&next) = self.transitions[current].get(&ch) {
                current = next;
            } else {
                return false;
            }
        }
        self.accept.contains(&current)
    }

    pub fn run(&self, input: &str) -> Vec<usize> {
        let mut path = vec![self.start];
        let mut current = self.start;
        for ch in input.chars() {
            if let Some(&next) = self.transitions[current].get(&ch) {
                current = next;
                path.push(current);
            } else {
                break;
            }
        }
        path
    }

    /// Minimize the DFA using Hopcroft's algorithm.
    pub fn minimize(&self) -> DFA {
        let mut partition: Vec<HashSet<usize>> = vec![
            self.accept.clone(),
            (0..self.states).filter(|s| !self.accept.contains(s)).collect(),
        ];

        loop {
            let mut new_partition = Vec::new();
            for group in &partition {
                let mut subgroups: HashMap<Vec<usize>, HashSet<usize>> = HashMap::new();
                for &state in group {
                    let signature: Vec<usize> = self.alphabet.iter().map(|&c| {
                        if let Some(&next) = self.transitions[state].get(&c) {
                            partition.iter().position(|g| g.contains(&next)).unwrap_or(usize::MAX)
                        } else {
                            usize::MAX
                        }
                    }).collect();
                    subgroups.entry(signature).or_default().insert(state);
                }
                new_partition.extend(subgroups.into_values());
            }

            if new_partition.len() == partition.len() { break; }
            partition = new_partition;
        }

        // Build minimized DFA
        let state_map: HashMap<usize, usize> = partition.iter().enumerate()
            .flat_map(|(i, group)| group.iter().map(move |&s| (s, i)))
            .collect();

        let new_states = partition.len();
        let new_start = state_map[&self.start];
        let new_accept: HashSet<usize> = self.accept.iter().map(|s| state_map[s]).collect();

        let mut new_dfa = DFA::new(new_states, self.alphabet.clone(), new_start, new_accept);
        for from in 0..self.states {
            for (&symbol, &to) in &self.transitions[from] {
                new_dfa.add_transition(state_map[&from], symbol, state_map[&to]);
            }
        }

        new_dfa
    }
}

/// Non-deterministic Finite Automaton.
#[derive(Debug, Clone)]
pub struct NFA {
    pub states: usize,
    pub alphabet: Vec<char>,
    pub transitions: Vec<HashMap<char, HashSet<usize>>>,
    pub epsilon_transitions: Vec<HashSet<usize>>,
    pub start: usize,
    pub accept: HashSet<usize>,
}

impl NFA {
    pub fn new(states: usize, alphabet: Vec<char>, start: usize, accept: HashSet<usize>) -> Self {
        Self {
            states,
            alphabet,
            transitions: vec![HashMap::new(); states],
            epsilon_transitions: vec![HashSet::new(); states],
            start,
            accept,
        }
    }

    pub fn add_transition(&mut self, from: usize, symbol: char, to: usize) {
        self.transitions[from].entry(symbol).or_default().insert(to);
    }

    pub fn add_epsilon(&mut self, from: usize, to: usize) {
        self.epsilon_transitions[from].insert(to);
    }

    pub fn epsilon_closure(&self, states: &HashSet<usize>) -> HashSet<usize> {
        let mut closure = states.clone();
        let mut queue: VecDeque<usize> = states.iter().copied().collect();

        while let Some(state) = queue.pop_front() {
            for &next in &self.epsilon_transitions[state] {
                if closure.insert(next) {
                    queue.push_back(next);
                }
            }
        }

        closure
    }

    pub fn accepts(&self, input: &str) -> bool {
        let mut current = self.epsilon_closure(&{
            let mut s = HashSet::new();
            s.insert(self.start);
            s
        });

        for ch in input.chars() {
            let mut next_states = HashSet::new();
            for &state in &current {
                if let Some(targets) = self.transitions[state].get(&ch) {
                    next_states.extend(targets);
                }
            }
            current = self.epsilon_closure(&next_states);
        }

        current.iter().any(|s| self.accept.contains(s))
    }

    /// Convert NFA to DFA using subset construction.
    pub fn to_dfa(&self) -> DFA {
        let start_set = self.epsilon_closure(&{
            let mut s = HashSet::new();
            s.insert(self.start);
            s
        });

        let mut state_map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut dfa_states = Vec::new();
        let mut queue = VecDeque::new();

        let start_key = sorted_vec(&start_set);
        state_map.insert(start_key.clone(), 0);
        dfa_states.push(start_set.clone());
        queue.push_back((0usize, start_set));

        let mut dfa = DFA::new(0, self.alphabet.clone(), 0, HashSet::new());

        while let Some((dfa_id, nfa_set)) = queue.pop_front() {
            for &symbol in &self.alphabet {
                let mut next_set = HashSet::new();
                for &state in &nfa_set {
                    if let Some(targets) = self.transitions[state].get(&symbol) {
                        next_set.extend(targets);
                    }
                }
                let next_set = self.epsilon_closure(&next_set);
                if next_set.is_empty() { continue; }

                let key = sorted_vec(&next_set);
                let next_id = if let Some(&id) = state_map.get(&key) {
                    id
                } else {
                    let id = dfa_states.len();
                    state_map.insert(key, id);
                    dfa_states.push(next_set.clone());
                    queue.push_back((id, next_set));
                    id
                };

                // Ensure DFA has enough states
                while dfa.states <= next_id.max(dfa_id) {
                    dfa.states += 1;
                    dfa.transitions.push(HashMap::new());
                }
                dfa.add_transition(dfa_id, symbol, next_id);
            }
        }

        // Set accept states
        for (i, nfa_set) in dfa_states.iter().enumerate() {
            if nfa_set.iter().any(|s| self.accept.contains(s)) {
                dfa.accept.insert(i);
            }
        }
        dfa.states = dfa_states.len();

        dfa
    }
}

fn sorted_vec(set: &HashSet<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = set.iter().copied().collect();
    v.sort();
    v
}

/// Pushdown Automaton (for context-free languages).
#[derive(Debug, Clone)]
pub struct PDA {
    pub states: usize,
    pub input_alphabet: Vec<char>,
    pub stack_alphabet: Vec<char>,
    pub transitions: Vec<HashMap<(char, char), Vec<(usize, Vec<char>)>>>,
    pub start: usize,
    pub accept: HashSet<usize>,
    pub start_stack: char,
}

impl PDA {
    pub fn new(
        states: usize,
        input_alphabet: Vec<char>,
        stack_alphabet: Vec<char>,
        start: usize,
        accept: HashSet<usize>,
        start_stack: char,
    ) -> Self {
        Self {
            states,
            input_alphabet,
            stack_alphabet,
            transitions: vec![HashMap::new(); states],
            start,
            accept,
            start_stack,
        }
    }

    pub fn add_transition(&mut self, from: usize, input: char, stack_top: char, to: usize, stack_push: Vec<char>) {
        self.transitions[from]
            .entry((input, stack_top))
            .or_default()
            .push((to, stack_push));
    }

    pub fn accepts(&self, input: &str) -> bool {
        let chars: Vec<char> = input.chars().collect();
        self.accepts_recursive(self.start, &chars, 0, vec![self.start_stack], 0)
    }

    fn accepts_recursive(
        &self,
        state: usize,
        input: &[char],
        pos: usize,
        mut stack: Vec<char>,
        depth: usize,
    ) -> bool {
        if depth > 1000 { return false; }

        if pos == input.len() && self.accept.contains(&state) {
            return true;
        }

        let stack_top = stack.last().copied();

        if let Some(top) = stack_top {
            // Try consuming input
            if pos < input.len() {
                if let Some(trans) = self.transitions[state].get(&(input[pos], top)) {
                    for &(next_state, ref push) in trans {
                        let mut new_stack = stack.clone();
                        new_stack.pop();
                        for &c in push.iter().rev() {
                            new_stack.push(c);
                        }
                        if self.accepts_recursive(next_state, input, pos + 1, new_stack, depth + 1) {
                            return true;
                        }
                    }
                }
            }

            // Try epsilon transitions
            if let Some(trans) = self.transitions[state].get(&('\0', top)) {
                for &(next_state, ref push) in trans {
                    let mut new_stack = stack.clone();
                    new_stack.pop();
                    for &c in push.iter().rev() {
                        new_stack.push(c);
                    }
                    if self.accepts_recursive(next_state, input, pos, new_stack, depth + 1) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// Turing Machine.
#[derive(Debug, Clone)]
pub struct TuringMachine {
    pub states: usize,
    pub tape_alphabet: Vec<char>,
    pub transitions: HashMap<(usize, char), (usize, char, Direction)>,
    pub start: usize,
    pub accept: usize,
    pub reject: usize,
    pub blank: char,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Left,
    Right,
    Stay,
}

impl TuringMachine {
    pub fn new(
        states: usize,
        tape_alphabet: Vec<char>,
        start: usize,
        accept: usize,
        reject: usize,
        blank: char,
    ) -> Self {
        Self {
            states,
            tape_alphabet,
            transitions: HashMap::new(),
            start,
            accept,
            reject,
            blank,
        }
    }

    pub fn add_transition(&mut self, state: usize, read: char, next: usize, write: char, dir: Direction) {
        self.transitions.insert((state, read), (next, write, dir));
    }

    pub fn run(&self, input: &str, max_steps: usize) -> TMResult {
        let mut tape: HashMap<i32, char> = input.chars().enumerate().map(|(i, c)| (i as i32, c)).collect();
        let mut head = 0i32;
        let mut state = self.start;

        for step in 0..max_steps {
            let symbol = tape.get(&head).copied().unwrap_or(self.blank);

            if let Some(&(next_state, write, dir)) = self.transitions.get(&(state, symbol)) {
                tape.insert(head, write);
                head += match dir {
                    Direction::Left => -1,
                    Direction::Right => 1,
                    Direction::Stay => 0,
                };
                state = next_state;

                if state == self.accept {
                    let tape_str = self.tape_to_string(&tape);
                    return TMResult { accepted: true, steps: step + 1, tape: tape_str };
                }
                if state == self.reject {
                    return TMResult { accepted: false, steps: step + 1, tape: String::new() };
                }
            } else {
                return TMResult { accepted: false, steps: 0, tape: String::new() };
            }
        }

        TMResult { accepted: false, steps: max_steps, tape: "timeout".to_string() }
    }

    fn tape_to_string(&self, tape: &HashMap<i32, char>) -> String {
        if tape.is_empty() { return String::new(); }
        let min = *tape.keys().min().unwrap();
        let max = *tape.keys().max().unwrap();
        (min..=max).map(|i| tape.get(&i).copied().unwrap_or(self.blank)).collect()
    }
}

#[derive(Debug)]
pub struct TMResult {
    pub accepted: bool,
    pub steps: usize,
    pub tape: String,
}

/// Regular expression to NFA (Thompson's construction).
pub fn regex_to_nfa(pattern: &str) -> NFA {
    let tokens: Vec<char> = pattern.chars().collect();
    let mut builder = NfaBuilder::new();
    builder.build(&tokens);
    builder.result()
}

struct NfaBuilder {
    nfa: NFA,
    state_count: usize,
}

impl NfaBuilder {
    fn new() -> Self {
        Self {
            nfa: NFA::new(0, vec![], 0, HashSet::new()),
            state_count: 0,
        }
    }

    fn new_state(&mut self) -> usize {
        let id = self.state_count;
        self.state_count += 1;
        self.nfa.states = self.state_count;
        self.nfa.transitions.push(HashMap::new());
        self.nfa.epsilon_transitions.push(HashSet::new());
        id
    }

    fn build(&mut self, pattern: &[char]) {
        let start = self.new_state();
        let accept = self.new_state();
        self.nfa.start = start;
        self.nfa.accept.insert(accept);

        self.nfa.add_epsilon(start, 1);
        let (end, _) = self.parse_expr(pattern, 0, 1, accept);
        self.nfa.add_epsilon(end, accept);
    }

    fn parse_expr(&mut self, pattern: &[char], pos: usize, from: usize, to: usize) -> (usize, usize) {
        if pos >= pattern.len() {
            return (from, pos);
        }

        let mut current_from = from;

        let (new_from, new_pos) = self.parse_term(pattern, pos, current_from, to);
        current_from = new_from;

        if new_pos < pattern.len() && pattern[new_pos] == '|' {
            let mid = self.new_state();
            self.nfa.add_epsilon(from, mid);
            let (end2, final_pos) = self.parse_expr(pattern, new_pos + 1, mid, to);
            return (end2, final_pos);
        }

        (current_from, new_pos)
    }

    fn parse_term(&mut self, pattern: &[char], pos: usize, from: usize, to: usize) -> (usize, usize) {
        let mut current_from = from;
        let mut current_pos = pos;

        while current_pos < pattern.len() && pattern[current_pos] != '|' && pattern[current_pos] != ')' {
            let (new_from, new_pos) = self.parse_factor(pattern, current_pos, current_from, to);
            current_from = new_from;
            current_pos = new_pos;
        }

        (current_from, current_pos)
    }

    fn parse_factor(&mut self, pattern: &[char], pos: usize, from: usize, to: usize) -> (usize, usize) {
        if pos >= pattern.len() { return (from, pos); }

        match pattern[pos] {
            '*' => {
                self.nfa.add_epsilon(from, to);
                (from, pos + 1)
            }
            '+' => {
                self.nfa.add_epsilon(to, from);
                (from, pos + 1)
            }
            '?' => {
                self.nfa.add_epsilon(from, to);
                (from, pos + 1)
            }
            '(' => {
                let (end, new_pos) = self.parse_expr(pattern, pos + 1, from, to);
                if new_pos < pattern.len() && pattern[new_pos] == ')' {
                    if new_pos + 1 < pattern.len() {
                        match pattern[new_pos + 1] {
                            '*' => {
                                self.nfa.add_epsilon(end, from);
                                self.nfa.add_epsilon(from, end);
                                return (from, new_pos + 2);
                            }
                            '+' => {
                                self.nfa.add_epsilon(end, from);
                                return (from, new_pos + 2);
                            }
                            '?' => {
                                self.nfa.add_epsilon(from, end);
                                return (from, new_pos + 2);
                            }
                            _ => {}
                        }
                    }
                    (end, new_pos + 1)
                } else {
                    (end, new_pos)
                }
            }
            c => {
                let mid = self.new_state();
                self.nfa.add_transition(from, c, mid);
                self.nfa.add_epsilon(mid, to);
                (to, pos + 1)
            }
        }
    }

    fn result(self) -> NFA {
        self.nfa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa() {
        let mut dfa = DFA::new(3, vec!['0', '1'], 0, {
            let mut s = HashSet::new();
            s.insert(2);
            s
        });
        dfa.add_transition(0, '0', 0);
        dfa.add_transition(0, '1', 1);
        dfa.add_transition(1, '0', 2);
        dfa.add_transition(1, '1', 0);
        dfa.add_transition(2, '0', 1);
        dfa.add_transition(2, '1', 2);

        assert!(dfa.accepts("10"));
        assert!(!dfa.accepts("11"));
    }

    #[test]
    fn test_nfa_to_dfa() {
        let mut nfa = NFA::new(3, vec!['a', 'b'], 0, {
            let mut s = HashSet::new();
            s.insert(2);
            s
        });
        nfa.add_transition(0, 'a', 0);
        nfa.add_transition(0, 'a', 1);
        nfa.add_transition(1, 'b', 2);

        let dfa = nfa.to_dfa();
        assert!(dfa.accepts("aab"));
        assert!(!dfa.accepts("aa"));
    }

    #[test]
    fn test_turing_machine() {
        // Palindrome checker
        let mut tm = TuringMachine::new(8, vec!['0', '1', 'B', 'X'], 0, 6, 7, 'B');
        // Simplified - just test basic operation
        tm.add_transition(0, '0', 1, 'X', Direction::Right);
        tm.add_transition(0, '1', 2, 'X', Direction::Right);
        tm.add_transition(0, 'B', 6, 'B', Direction::Stay); // accept empty
        tm.add_transition(0, 'X', 6, 'X', Direction::Stay); // accept single

        let result = tm.run("", 100);
        assert!(result.accepted);
    }

    #[test]
    fn test_pda() {
        let mut pda = PDA::new(3, vec!['a', 'b'], vec!['Z', 'A'], 0, {
            let mut s = HashSet::new();
            s.insert(2);
            s
        }, 'Z');

        // a^n b^n
        pda.add_transition(0, 'a', 'Z', 0, vec!['A', 'Z']);
        pda.add_transition(0, 'a', 'A', 0, vec!['A', 'A']);
        pda.add_transition(0, 'b', 'A', 1, vec![]);
        pda.add_transition(1, 'b', 'A', 1, vec![]);
        pda.add_transition(1, '\0', 'Z', 2, vec!['Z']);

        assert!(pda.accepts("aabb"));
        assert!(pda.accepts("aaaabbbb"));
        assert!(!pda.accepts("aab"));
    }
}
