/// Consensus protocol abstractions.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ConsensusMessage {
    Prepare { n: u64 },
    Promise { n: u64, value: Option<String> },
    Accept { n: u64, value: String },
    Accepted { n: u64 },
    Decide { value: String },
}

#[derive(Debug, Clone)]
pub struct ConsensusNode {
    pub id: u64,
    pub proposal_number: u64,
    pub accepted_number: u64,
    pub accepted_value: Option<String>,
    pub promised_number: u64,
    pub peers: Vec<u64>,
    pub promises_received: u64,
    pub accepts_received: u64,
    pub decided_value: Option<String>,
}

impl ConsensusNode {
    pub fn new(id: u64, peers: Vec<u64>) -> Self {
        Self {
            id,
            proposal_number: 0,
            accepted_number: 0,
            accepted_value: None,
            promised_number: 0,
            peers,
            promises_received: 0,
            accepts_received: 0,
            decided_value: None,
        }
    }

    pub fn prepare(&mut self) -> ConsensusMessage {
        self.proposal_number += 1;
        self.promises_received = 0;
        ConsensusMessage::Prepare { n: self.proposal_number }
    }

    pub fn handle_prepare(&mut self, n: u64) -> ConsensusMessage {
        if n > self.promised_number {
            self.promised_number = n;
            ConsensusMessage::Promise {
                n,
                value: self.accepted_value.clone(),
            }
        } else {
            ConsensusMessage::Promise {
                n: self.promised_number,
                value: None,
            }
        }
    }

    pub fn handle_promise(&mut self, n: u64, value: Option<String>) -> Option<ConsensusMessage> {
        if n != self.proposal_number {
            return None;
        }

        self.promises_received += 1;

        if let Some(v) = value {
            // Adopt highest accepted value
            self.accepted_value = Some(v);
        }

        if self.promises_received > self.peers.len() as u64 / 2 {
            let value = self.accepted_value.clone().unwrap_or_default();
            Some(ConsensusMessage::Accept {
                n: self.proposal_number,
                value,
            })
        } else {
            None
        }
    }

    pub fn handle_accept(&mut self, n: u64, value: &str) -> ConsensusMessage {
        if n >= self.promised_number {
            self.accepted_number = n;
            self.accepted_value = Some(value.to_string());
            ConsensusMessage::Accepted { n }
        } else {
            ConsensusMessage::Accepted { n: 0 }
        }
    }

    pub fn handle_accepted(&mut self, n: u64) -> Option<ConsensusMessage> {
        if n != self.proposal_number {
            return None;
        }

        self.accepts_received += 1;

        if self.accepts_received > self.peers.len() as u64 / 2 {
            self.decided_value = self.accepted_value.clone();
            Some(ConsensusMessage::Decide {
                value: self.accepted_value.clone().unwrap_or_default(),
            })
        } else {
            None
        }
    }

    pub fn is_decided(&self) -> bool {
        self.decided_value.is_some()
    }
}

/// Vector clocks for causal ordering
#[derive(Debug, Clone)]
pub struct VectorClock {
    clocks: HashMap<u64, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: u64) {
        *self.clocks.entry(node_id).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (&node, &time) in &other.clocks {
            let entry = self.clocks.entry(node).or_insert(0);
            *entry = (*entry).max(time);
        }
    }

    pub fn get(&self, node_id: u64) -> u64 {
        *self.clocks.get(&node_id).unwrap_or(&0)
    }

    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut all_leq = true;
        let mut any_lt = false;

        for (&node, &time) in &self.clocks {
            let other_time = other.get(node);
            if time > other_time {
                all_leq = false;
                break;
            }
            if time < other_time {
                any_lt = true;
            }
        }

        all_leq && any_lt
    }

    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_prepare() {
        let mut node = ConsensusNode::new(1, vec![2, 3]);
        let msg = node.prepare();
        assert!(matches!(msg, ConsensusMessage::Prepare { n: 1 }));
    }

    #[test]
    fn test_consensus_flow() {
        let mut proposer = ConsensusNode::new(1, vec![2, 3]);
        let mut acceptor1 = ConsensusNode::new(2, vec![1, 3]);
        let mut acceptor2 = ConsensusNode::new(3, vec![1, 2]);

        // Prepare
        let prepare = proposer.prepare();

        // Handle promises
        let promise1 = acceptor1.handle_prepare(1);
        let promise2 = acceptor2.handle_prepare(1);

        // Process promises
        let accept = proposer.handle_promise(1, None);
        assert!(accept.is_none()); // Need majority

        let accept = proposer.handle_promise(1, None);
        assert!(accept.is_some()); // Got majority
    }

    #[test]
    fn test_vector_clock() {
        let mut vc1 = VectorClock::new();
        vc1.increment(1);
        vc1.increment(1);

        let mut vc2 = VectorClock::new();
        vc2.increment(2);

        assert_eq!(vc1.get(1), 2);
        assert_eq!(vc2.get(2), 1);
    }

    #[test]
    fn test_happens_before() {
        let mut vc1 = VectorClock::new();
        vc1.increment(1);

        let mut vc2 = VectorClock::new();
        vc2.increment(1);
        vc2.increment(2);

        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_concurrent() {
        let mut vc1 = VectorClock::new();
        vc1.increment(1);

        let mut vc2 = VectorClock::new();
        vc2.increment(2);

        assert!(vc1.concurrent_with(&vc2));
    }
}
