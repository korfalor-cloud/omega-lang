/// Distributed consensus: Raft, Paxos simulation, vector clocks.

use std::collections::HashMap;

/// Vector clock for causal ordering.
#[derive(Debug, Clone)]
pub struct VectorClock {
    pub clocks: HashMap<usize, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self { clocks: HashMap::new() }
    }

    pub fn increment(&mut self, node_id: usize) {
        *self.clocks.entry(node_id).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (&node, &time) in &other.clocks {
            let entry = self.clocks.entry(node).or_insert(0);
            *entry = (*entry).max(time);
        }
    }

    pub fn get(&self, node_id: usize) -> u64 {
        self.clocks.get(&node_id).copied().unwrap_or(0)
    }

    /// Check if self happens before other.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;
        for (&node, &time) in &self.clocks {
            let other_time = other.clocks.get(&node).copied().unwrap_or(0);
            if time > other_time { return false; }
            if time < other_time { strictly_less = true; }
        }
        strictly_less || self.clocks.len() < other.clocks.len()
    }

    /// Check if concurrent (neither happens before the other).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

/// Lamport timestamp.
#[derive(Debug, Clone)]
pub struct LamportClock {
    pub time: u64,
}

impl LamportClock {
    pub fn new() -> Self {
        Self { time: 0 }
    }

    pub fn increment(&mut self) {
        self.time += 1;
    }

    pub fn receive(&mut self, remote_time: u64) {
        self.time = self.time.max(remote_time) + 1;
    }
}

/// Raft consensus simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
pub struct RaftNode {
    pub id: usize,
    pub state: RaftState,
    pub current_term: u64,
    pub voted_for: Option<usize>,
    pub log: Vec<LogEntry>,
    pub commit_index: usize,
    pub last_applied: usize,
    pub next_index: HashMap<usize, usize>,
    pub match_index: HashMap<usize, usize>,
    pub election_timeout: u64,
    pub heartbeat_timer: u64,
    pub peers: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,
    pub command: String,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub enum RaftMessage {
    RequestVote { term: u64, candidate_id: usize, last_log_index: usize, last_log_term: u64 },
    RequestVoteResponse { term: u64, vote_granted: bool },
    AppendEntries { term: u64, leader_id: usize, prev_log_index: usize, prev_log_term: u64, entries: Vec<LogEntry>, leader_commit: usize },
    AppendEntriesResponse { term: u64, success: bool, match_index: usize },
}

impl RaftNode {
    pub fn new(id: usize, peers: Vec<usize>) -> Self {
        Self {
            id,
            state: RaftState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            election_timeout: 150,
            heartbeat_timer: 0,
            peers,
        }
    }

    pub fn handle_message(&mut self, msg: RaftMessage) -> Vec<(usize, RaftMessage)> {
        let mut responses = Vec::new();

        match msg {
            RaftMessage::RequestVote { term, candidate_id, last_log_index, last_log_term } => {
                if term < self.current_term {
                    responses.push((candidate_id, RaftMessage::RequestVoteResponse {
                        term: self.current_term,
                        vote_granted: false,
                    }));
                    return responses;
                }

                if term > self.current_term {
                    self.current_term = term;
                    self.state = RaftState::Follower;
                    self.voted_for = None;
                }

                let vote_granted = self.voted_for.is_none() || self.voted_for == Some(candidate_id);
                let log_ok = last_log_term > self.last_log_term()
                    || (last_log_term == self.last_log_term() && last_log_index >= self.log.len());

                if vote_granted && log_ok {
                    self.voted_for = Some(candidate_id);
                    self.heartbeat_timer = 0;
                }

                responses.push((candidate_id, RaftMessage::RequestVoteResponse {
                    term: self.current_term,
                    vote_granted: vote_granted && log_ok,
                }));
            }

            RaftMessage::RequestVoteResponse { term, vote_granted } => {
                if term > self.current_term {
                    self.current_term = term;
                    self.state = RaftState::Follower;
                    self.voted_for = None;
                    return responses;
                }

                if self.state == RaftState::Candidate && vote_granted {
                    // Count votes (simplified)
                    // In a real implementation, would track vote count
                }
            }

            RaftMessage::AppendEntries { term, leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                if term < self.current_term {
                    responses.push((leader_id, RaftMessage::AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        match_index: 0,
                    }));
                    return responses;
                }

                self.current_term = term;
                self.state = RaftState::Follower;
                self.heartbeat_timer = 0;

                // Check log consistency
                if prev_log_index > 0 {
                    if prev_log_index > self.log.len() {
                        responses.push((leader_id, RaftMessage::AppendEntriesResponse {
                            term: self.current_term,
                            success: false,
                            match_index: 0,
                        }));
                        return responses;
                    }
                    if self.log[prev_log_index - 1].term != prev_log_term {
                        responses.push((leader_id, RaftMessage::AppendEntriesResponse {
                            term: self.current_term,
                            success: false,
                            match_index: 0,
                        }));
                        return responses;
                    }
                }

                // Append entries
                for entry in entries {
                    if entry.index <= self.log.len() {
                        self.log.truncate(entry.index - 1);
                    }
                    self.log.push(entry);
                }

                // Update commit index
                if leader_commit > self.commit_index {
                    self.commit_index = leader_commit.min(self.log.len());
                }

                responses.push((leader_id, RaftMessage::AppendEntriesResponse {
                    term: self.current_term,
                    success: true,
                    match_index: self.log.len(),
                }));
            }

            RaftMessage::AppendEntriesResponse { term, success, match_index } => {
                if term > self.current_term {
                    self.current_term = term;
                    self.state = RaftState::Follower;
                    self.voted_for = None;
                    return responses;
                }

                if self.state == RaftState::Leader && success {
                    // Update next_index and match_index for the follower
                    // (simplified - would need to track which node sent the response)
                }
            }
        }

        responses
    }

    pub fn start_election(&mut self) -> Vec<(usize, RaftMessage)> {
        self.current_term += 1;
        self.state = RaftState::Candidate;
        self.voted_for = Some(self.id);
        self.heartbeat_timer = 0;

        let last_log_index = self.log.len();
        let last_log_term = self.last_log_term();

        self.peers.iter().map(|&peer| {
            (peer, RaftMessage::RequestVote {
                term: self.current_term,
                candidate_id: self.id,
                last_log_index,
                last_log_term,
            })
        }).collect()
    }

    pub fn append_entries(&mut self, command: String) {
        if self.state != RaftState::Leader { return; }

        let index = self.log.len() + 1;
        self.log.push(LogEntry {
            term: self.current_term,
            command,
            index,
        });
    }

    pub fn send_heartbeats(&self) -> Vec<(usize, RaftMessage)> {
        if self.state != RaftState::Leader { return Vec::new(); }

        self.peers.iter().map(|&peer| {
            let next_idx = self.next_index.get(&peer).copied().unwrap_or(1);
            let prev_log_index = next_idx.saturating_sub(1);
            let prev_log_term = if prev_log_index > 0 && prev_log_index <= self.log.len() {
                self.log[prev_log_index - 1].term
            } else {
                0
            };

            (peer, RaftMessage::AppendEntries {
                term: self.current_term,
                leader_id: self.id,
                prev_log_index,
                prev_log_term,
                entries: self.log[next_idx.saturating_sub(1)..].to_vec(),
                leader_commit: self.commit_index,
            })
        }).collect()
    }

    fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }
}

/// Simple Paxos simulation.
pub struct PaxosNode {
    pub id: usize,
    pub proposal_number: u64,
    pub accepted_proposal: Option<u64>,
    pub accepted_value: Option<String>,
    pub promised_proposal: Option<u64>,
    pub peers: Vec<usize>,
}

impl PaxosNode {
    pub fn new(id: usize, peers: Vec<usize>) -> Self {
        Self {
            id,
            proposal_number: 0,
            accepted_proposal: None,
            accepted_value: None,
            promised_proposal: None,
            peers,
        }
    }

    pub fn prepare(&mut self, proposal_number: u64) -> PaxosMessage {
        self.proposal_number = proposal_number;
        PaxosMessage::Prepare {
            proposal_number,
            proposer_id: self.id,
        }
    }

    pub fn handle_prepare(&mut self, proposal_number: u64, proposer_id: usize) -> PaxosMessage {
        if self.promised_proposal.is_none() || proposal_number > self.promised_proposal.unwrap() {
            self.promised_proposal = Some(proposal_number);
            PaxosMessage::Promise {
                proposal_number,
                accepted_proposal: self.accepted_proposal,
                accepted_value: self.accepted_value.clone(),
                responder_id: self.id,
            }
        } else {
            PaxosMessage::Nack {
                proposal_number,
                promised_proposal: self.promised_proposal.unwrap(),
                responder_id: self.id,
            }
        }
    }

    pub fn accept(&mut self, proposal_number: u64, value: String) -> PaxosMessage {
        PaxosMessage::Accept {
            proposal_number,
            value,
            proposer_id: self.id,
        }
    }

    pub fn handle_accept(&mut self, proposal_number: u64, value: String, proposer_id: usize) -> PaxosMessage {
        if self.promised_proposal.is_none() || proposal_number >= self.promised_proposal.unwrap() {
            self.promised_proposal = Some(proposal_number);
            self.accepted_proposal = Some(proposal_number);
            self.accepted_value = Some(value.clone());
            PaxosMessage::Accepted {
                proposal_number,
                value,
                acceptor_id: self.id,
            }
        } else {
            PaxosMessage::Nack {
                proposal_number,
                promised_proposal: self.promised_proposal.unwrap(),
                responder_id: self.id,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum PaxosMessage {
    Prepare { proposal_number: u64, proposer_id: usize },
    Promise { proposal_number: u64, accepted_proposal: Option<u64>, accepted_value: Option<String>, responder_id: usize },
    Accept { proposal_number: u64, value: String, proposer_id: usize },
    Accepted { proposal_number: u64, value: String, acceptor_id: usize },
    Nack { proposal_number: u64, promised_proposal: u64, responder_id: usize },
}

/// CRDT: G-Counter (grow-only counter).
#[derive(Debug, Clone)]
pub struct GCounter {
    pub counts: HashMap<usize, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }

    pub fn increment(&mut self, node_id: usize) {
        *self.counts.entry(node_id).or_insert(0) += 1;
    }

    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn merge(&mut self, other: &GCounter) {
        for (&node, &count) in &other.counts {
            let entry = self.counts.entry(node).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

/// CRDT: PN-Counter (positive-negative counter).
#[derive(Debug, Clone)]
pub struct PNCounter {
    pub positive: GCounter,
    pub negative: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self { positive: GCounter::new(), negative: GCounter::new() }
    }

    pub fn increment(&mut self, node_id: usize) {
        self.positive.increment(node_id);
    }

    pub fn decrement(&mut self, node_id: usize) {
        self.negative.increment(node_id);
    }

    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

/// CRDT: G-Set (grow-only set).
#[derive(Debug, Clone)]
pub struct GSet<T: Eq + std::hash::Hash + Clone> {
    pub elements: std::collections::HashSet<T>,
}

impl<T: Eq + std::hash::Hash + Clone> GSet<T> {
    pub fn new() -> Self {
        Self { elements: std::collections::HashSet::new() }
    }

    pub fn add(&mut self, element: T) {
        self.elements.insert(element);
    }

    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    pub fn merge(&mut self, other: &GSet<T>) {
        self.elements.extend(other.elements.iter().cloned());
    }

    pub fn value(&self) -> &std::collections::HashSet<T> {
        &self.elements
    }
}

/// CRDT: LWW-Register (Last-Writer-Wins Register).
#[derive(Debug, Clone)]
pub struct LWWRegister<T: Clone> {
    pub value: T,
    pub timestamp: u64,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new(value: T, timestamp: u64) -> Self {
        Self { value, timestamp }
    }

    pub fn set(&mut self, value: T, timestamp: u64) {
        if timestamp > self.timestamp {
            self.value = value;
            self.timestamp = timestamp;
        }
    }

    pub fn merge(&mut self, other: &LWWRegister<T>) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
        }
    }

    pub fn get(&self) -> &T {
        &self.value
    }
}

/// CRDT: OR-Set (Observed-Remove Set).
#[derive(Debug, Clone)]
pub struct ORSet<T: Eq + std::hash::Hash + Clone> {
    pub elements: HashMap<T, Vec<u64>>,
    pub tombstones: HashMap<T, Vec<u64>>,
    pub counter: u64,
}

impl<T: Eq + std::hash::Hash + Clone> ORSet<T> {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashMap::new(),
            counter: 0,
        }
    }

    pub fn add(&mut self, element: T) {
        self.counter += 1;
        self.elements.entry(element).or_default().push(self.counter);
    }

    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.remove(element) {
            self.tombstones.entry(element.clone()).or_default().extend(tags);
        }
    }

    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains_key(element)
    }

    pub fn merge(&mut self, other: &ORSet<T>) {
        for (element, tags) in &other.elements {
            let entry = self.elements.entry(element.clone()).or_default();
            for tag in tags {
                if !entry.contains(tag) && !self.tombstones.get(element).map_or(false, |t| t.contains(tag)) {
                    entry.push(*tag);
                }
            }
        }
        for (element, tags) in &other.tombstones {
            let entry = self.tombstones.entry(element.clone()).or_default();
            for tag in tags {
                if !entry.contains(tag) {
                    entry.push(*tag);
                }
            }
        }
        self.counter = self.counter.max(other.counter);
    }

    pub fn value(&self) -> Vec<&T> {
        self.elements.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock() {
        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();

        vc1.increment(1);
        vc1.increment(1);
        vc2.increment(2);

        assert!(vc1.get(1) == 2);
        assert!(vc2.get(2) == 1);
        assert!(vc1.is_concurrent(&vc2));
    }

    #[test]
    fn test_g_counter() {
        let mut c1 = GCounter::new();
        let mut c2 = GCounter::new();

        c1.increment(1);
        c1.increment(1);
        c2.increment(2);
        c2.increment(2);
        c2.increment(2);

        c1.merge(&c2);
        assert_eq!(c1.value(), 5);
    }

    #[test]
    fn test_pn_counter() {
        let mut counter = PNCounter::new();
        counter.increment(1);
        counter.increment(1);
        counter.decrement(1);
        assert_eq!(counter.value(), 1);
    }

    #[test]
    fn test_or_set() {
        let mut s1 = ORSet::new();
        s1.add("a");
        s1.add("b");

        let mut s2 = s1.clone();
        s2.remove(&"a");
        s2.add("c");

        s1.merge(&s2);
        assert!(s1.contains(&"b"));
        assert!(s1.contains(&"c"));
        assert!(!s1.contains(&"a"));
    }

    #[test]
    fn test_lww_register() {
        let mut reg = LWWRegister::new(0, 0);
        reg.set(1, 10);
        reg.set(2, 5);
        assert_eq!(*reg.get(), 1);
    }
}
