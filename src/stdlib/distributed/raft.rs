/// Raft consensus algorithm implementation.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
    pub committed: bool,
}

#[derive(Debug, Clone)]
pub struct RaftNode {
    pub id: u64,
    pub state: NodeState,
    pub current_term: u64,
    pub voted_for: Option<u64>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub peers: Vec<u64>,
    // Leader state
    pub next_index: HashMap<u64, u64>,
    pub match_index: HashMap<u64, u64>,
    // Election state
    pub votes_received: u64,
    pub election_timeout: u64,
    pub heartbeat_timeout: u64,
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: u64,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

#[derive(Debug, Clone)]
pub struct VoteRequest {
    pub term: u64,
    pub candidate_id: u64,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

#[derive(Debug, Clone)]
pub struct VoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}

impl RaftNode {
    pub fn new(id: u64, peers: Vec<u64>) -> Self {
        Self {
            id,
            state: NodeState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            peers,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            votes_received: 0,
            election_timeout: 150,
            heartbeat_timeout: 50,
            last_heartbeat: 0,
        }
    }

    pub fn start_election(&mut self) {
        self.state = NodeState::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.votes_received = 1;
    }

    pub fn handle_vote_request(&mut self, request: &VoteRequest) -> VoteResponse {
        if request.term < self.current_term {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        if request.term > self.current_term {
            self.current_term = request.term;
            self.state = NodeState::Follower;
            self.voted_for = None;
        }

        let vote_granted = self.voted_for.is_none() || self.voted_for == Some(request.candidate_id);

        if vote_granted {
            self.voted_for = Some(request.candidate_id);
        }

        VoteResponse {
            term: self.current_term,
            vote_granted,
        }
    }

    pub fn handle_vote_response(&mut self, response: &VoteResponse) {
        if response.term > self.current_term {
            self.current_term = response.term;
            self.state = NodeState::Follower;
            return;
        }

        if self.state != NodeState::Candidate {
            return;
        }

        if response.vote_granted {
            self.votes_received += 1;
            if self.votes_received > (self.peers.len() as u64 + 1) / 2 {
                self.become_leader();
            }
        }
    }

    fn become_leader(&mut self) {
        self.state = NodeState::Leader;
        for &peer in &self.peers {
            self.next_index.insert(peer, self.log.len() as u64 + 1);
            self.match_index.insert(peer, 0);
        }
    }

    pub fn append_entries(&mut self, request: &AppendEntriesRequest) -> AppendEntriesResponse {
        if request.term < self.current_term {
            return AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
            };
        }

        if request.term > self.current_term {
            self.current_term = request.term;
            self.state = NodeState::Follower;
            self.voted_for = None;
        }

        self.last_heartbeat = 0;

        // Check log consistency
        if request.prev_log_index > 0 {
            if request.prev_log_index > self.log.len() as u64 {
                return AppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.log.len() as u64,
                };
            }
            let prev_entry = &self.log[request.prev_log_index as usize - 1];
            if prev_entry.term != request.prev_log_term {
                return AppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: 0,
                };
            }
        }

        // Append entries
        for entry in &request.entries {
            let idx = entry.index as usize - 1;
            if idx < self.log.len() {
                if self.log[idx].term != entry.term {
                    self.log.truncate(idx);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Update commit index
        if request.leader_commit > self.commit_index {
            self.commit_index = request.leader_commit.min(self.log.len() as u64);
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            match_index: self.log.len() as u64,
        }
    }

    pub fn propose(&mut self, command: &str) -> Option<LogEntry> {
        if self.state != NodeState::Leader {
            return None;
        }

        let entry = LogEntry {
            term: self.current_term,
            index: self.log.len() as u64 + 1,
            command: command.to_string(),
            committed: false,
        };
        self.log.push(entry.clone());
        Some(entry)
    }

    pub fn commit(&mut self) {
        // In a real implementation, this would check match_index from majority
        for entry in &mut self.log {
            if entry.index <= self.commit_index {
                entry.committed = true;
            }
        }
    }

    pub fn tick(&mut self) {
        self.last_heartbeat += 1;

        if self.state == NodeState::Follower && self.last_heartbeat >= self.election_timeout {
            self.start_election();
        }

        if self.state == NodeState::Leader && self.last_heartbeat >= self.heartbeat_timeout {
            self.last_heartbeat = 0;
            // Would send heartbeats to peers
        }
    }

    pub fn is_leader(&self) -> bool {
        self.state == NodeState::Leader
    }

    pub fn log_length(&self) -> usize {
        self.log.len()
    }

    pub fn committed_entries(&self) -> Vec<&LogEntry> {
        self.log.iter().filter(|e| e.committed).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = RaftNode::new(1, vec![2, 3]);
        assert_eq!(node.state, NodeState::Follower);
        assert_eq!(node.current_term, 0);
    }

    #[test]
    fn test_election() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        node.start_election();
        assert_eq!(node.state, NodeState::Candidate);
        assert_eq!(node.current_term, 1);
        assert_eq!(node.votes_received, 1);
    }

    #[test]
    fn test_vote_request() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        let request = VoteRequest {
            term: 1,
            candidate_id: 2,
            last_log_index: 0,
            last_log_term: 0,
        };
        let response = node.handle_vote_request(&request);
        assert!(response.vote_granted);
        assert_eq!(node.voted_for, Some(2));
    }

    #[test]
    fn test_become_leader() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        node.start_election();
        node.handle_vote_response(&VoteResponse { term: 1, vote_granted: true });
        node.handle_vote_response(&VoteResponse { term: 1, vote_granted: true });
        assert_eq!(node.state, NodeState::Leader);
    }

    #[test]
    fn test_propose() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        node.become_leader();
        let entry = node.propose("SET x=1");
        assert!(entry.is_some());
        assert_eq!(node.log_length(), 1);
    }

    #[test]
    fn test_append_entries() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        let request = AppendEntriesRequest {
            term: 1,
            leader_id: 2,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                command: "SET x=1".to_string(),
                committed: false,
            }],
            leader_commit: 0,
        };
        let response = node.append_entries(&request);
        assert!(response.success);
        assert_eq!(node.log_length(), 1);
    }

    #[test]
    fn test_tick() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        for _ in 0..150 {
            node.tick();
        }
        assert_eq!(node.state, NodeState::Candidate);
    }
}
