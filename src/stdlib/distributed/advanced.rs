/// Advanced distributed systems primitives.
///
/// Contains consistent hashing, vector clocks, CRDTs (G-Counter, PN-Counter,
/// OR-Set), a lightweight Raft consensus simulation, and a distributed hash
/// table built on top of the consistent-hash ring.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Consistent hashing
// ---------------------------------------------------------------------------

const VIRTUAL_NODES: u64 = 128;

#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, String>,
    nodes: HashSet<String>,
}

impl ConsistentHashRing {
    pub fn new() -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: HashSet::new(),
        }
    }

    fn hash_key(key: &str) -> u64 {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }

    pub fn add_node(&mut self, node: &str) {
        self.nodes.insert(node.to_string());
        for i in 0..VIRTUAL_NODES {
            let vnode_key = format!("{}#{}", node, i);
            let h = Self::hash_key(&vnode_key);
            self.ring.insert(h, node.to_string());
        }
    }

    pub fn remove_node(&mut self, node: &str) {
        self.nodes.remove(node);
        for i in 0..VIRTUAL_NODES {
            let vnode_key = format!("{}#{}", node, i);
            let h = Self::hash_key(&vnode_key);
            self.ring.remove(&h);
        }
    }

    pub fn get_node(&self, key: &str) -> Option<String> {
        if self.ring.is_empty() {
            return None;
        }
        let h = Self::hash_key(key);
        // First entry >= h, or wrap around to the first entry.
        let target = self
            .ring
            .range(h..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, v)| v.clone());
        target
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Vector clocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvancedVectorClock {
    clocks: HashMap<String, u64>,
}

impl AdvancedVectorClock {
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &AdvancedVectorClock) {
        for (node, &time) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(time);
        }
    }

    pub fn get(&self, node_id: &str) -> u64 {
        *self.clocks.get(node_id).unwrap_or(&0)
    }

    /// Returns `true` if `self` happened-before `other`.
    pub fn happens_before(&self, other: &AdvancedVectorClock) -> bool {
        let mut all_leq = true;
        let mut any_lt = false;

        for (node, &time) in &self.clocks {
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

    /// Returns `true` if neither clock happens-before the other.
    pub fn concurrent_with(&self, other: &AdvancedVectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

// ---------------------------------------------------------------------------
// CRDTs
// ---------------------------------------------------------------------------

/// Grow-only counter -- each node maintains its own monotonic sub-counter.
#[derive(Debug, Clone)]
pub struct GCounter {
    counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: &str, delta: u64) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += delta;
    }

    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn merge(&mut self, other: &GCounter) {
        for (node, &count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

/// Positive-Negative counter built from two G-Counters.
#[derive(Debug, Clone)]
pub struct PNCounter {
    positive: GCounter,
    negative: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self {
            positive: GCounter::new(),
            negative: GCounter::new(),
        }
    }

    pub fn increment(&mut self, node_id: &str, delta: i64) {
        if delta >= 0 {
            self.positive.increment(node_id, delta as u64);
        } else {
            self.negative.increment(node_id, (-delta) as u64);
        }
    }

    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

/// Observed-Remove Set -- elements can be added and removed; removes are
/// tracked per-element so that concurrent add/remove both survive.
#[derive(Debug, Clone)]
pub struct ORSet {
    elements: HashMap<String, HashSet<String>>,
    tombstones: HashMap<String, HashSet<String>>,
}

impl ORSet {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashMap::new(),
        }
    }

    pub fn add(&mut self, element: &str, tag: &str) {
        self.elements
            .entry(element.to_string())
            .or_default()
            .insert(tag.to_string());
    }

    pub fn remove(&mut self, element: &str) {
        if let Some(tags) = self.elements.remove(element) {
            self.tombstones
                .entry(element.to_string())
                .or_default()
                .extend(tags);
        }
    }

    pub fn contains(&self, element: &str) -> bool {
        self.elements
            .get(element)
            .map(|tags| !tags.is_empty())
            .unwrap_or(false)
    }

    pub fn members(&self) -> Vec<String> {
        self.elements
            .iter()
            .filter(|(_, tags)| !tags.is_empty())
            .map(|(k, _)| k.clone())
            .collect()
    }

    pub fn merge(&mut self, other: &ORSet) {
        for (elem, tags) in &other.elements {
            let entry = self.elements.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
        for (elem, tags) in &other.tombstones {
            let entry = self.tombstones.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
        // Remove elements whose tags were tombstoned.
        for (elem, tomb) in &self.tombstones {
            if let Some(tags) = self.elements.get_mut(elem) {
                tags.retain(|t| !tomb.contains(t));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.elements
            .values()
            .map(|tags| tags.len())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Raft consensus simulation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
pub struct RaftSimNode {
    pub id: String,
    pub role: RaftRole,
    pub term: u64,
    pub log: Vec<(u64, String)>,
    pub commit_idx: usize,
    votes_received: HashSet<String>,
    pub peers: Vec<String>,
}

impl RaftSimNode {
    pub fn new(id: &str, peers: Vec<String>) -> Self {
        Self {
            id: id.to_string(),
            role: RaftRole::Follower,
            term: 0,
            log: Vec::new(),
            commit_idx: 0,
            votes_received: HashSet::new(),
            peers,
        }
    }

    pub fn start_election(&mut self) {
        self.role = RaftRole::Candidate;
        self.term += 1;
        self.votes_received.clear();
        self.votes_received.insert(self.id.clone());
    }

    /// Returns `true` if the vote is granted.
    pub fn request_vote(&mut self, candidate_id: &str, candidate_term: u64) -> bool {
        if candidate_term > self.term {
            self.term = candidate_term;
            self.role = RaftRole::Follower;
            true
        } else {
            false
        }
    }

    pub fn record_vote(&mut self, voter_id: &str) {
        self.votes_received.insert(voter_id.to_string());
    }

    pub fn has_majority(&self) -> bool {
        let total = self.peers.len() + 1;
        self.votes_received.len() > total / 2
    }

    pub fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
    }

    pub fn append_entry(&mut self, command: &str) {
        let idx = self.log.len() as u64 + 1;
        self.log.push((self.term, command.to_string()));
        let _ = idx;
    }

    pub fn commit(&mut self) {
        self.commit_idx = self.log.len();
    }

    pub fn apply_log(&self) -> Vec<&str> {
        self.log[..self.commit_idx]
            .iter()
            .map(|(_, cmd)| cmd.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Distributed Hash Table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DistributedHashTable {
    ring: ConsistentHashRing,
    store: HashMap<String, String>,
    replication_factor: usize,
}

impl DistributedHashTable {
    pub fn new(replication_factor: usize) -> Self {
        Self {
            ring: ConsistentHashRing::new(),
            store: HashMap::new(),
            replication_factor: replication_factor.max(1),
        }
    }

    pub fn add_node(&mut self, node: &str) {
        self.ring.add_node(node);
    }

    pub fn put(&mut self, key: &str, value: &str) -> Vec<String> {
        let mut nodes = self.get_nodes_for_key(key);
        // Primary stores the value.
        if let Some(primary) = nodes.first() {
            self.store
                .insert(format!("{}@{}", primary, key), value.to_string());
        }
        nodes
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let nodes = self.get_nodes_for_key(key);
        for node in &nodes {
            let store_key = format!("{}@{}", node, key);
            if let Some(val) = self.store.get(&store_key) {
                return Some(val.clone());
            }
        }
        None
    }

    pub fn delete(&mut self, key: &str) -> bool {
        let nodes = self.get_nodes_for_key(key);
        let mut deleted = false;
        for node in &nodes {
            let store_key = format!("{}@{}", node, key);
            if self.store.remove(&store_key).is_some() {
                deleted = true;
            }
        }
        deleted
    }

    fn get_nodes_for_key(&self, key: &str) -> Vec<String> {
        // Walk the ring starting from the key's position, collecting unique
        // physical nodes until we reach the replication factor.
        if self.ring.node_count() == 0 {
            return Vec::new();
        }
        let h = ConsistentHashRing::hash_key(key);
        let mut result: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // Forward iteration from h.
        for (_, node) in self.ring.ring.range(h..) {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() >= self.replication_factor {
                    return result;
                }
            }
        }
        // Wrap around.
        for (_, node) in self.ring.ring.range(..h) {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() >= self.replication_factor {
                    return result;
                }
            }
        }
        result
    }

    pub fn node_count(&self) -> usize {
        self.ring.node_count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Consistent hashing -------------------------------------------------

    #[test]
    fn test_hash_ring_basic() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("A");
        ring.add_node("B");
        ring.add_node("C");
        assert_eq!(ring.node_count(), 3);
        let node = ring.get_node("some-key").unwrap();
        assert!(["A", "B", "C"].contains(&node.as_str()));
    }

    #[test]
    fn test_hash_ring_remove() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("A");
        ring.add_node("B");
        ring.remove_node("A");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.get_node("k").unwrap(), "B");
    }

    // -- Vector clocks ------------------------------------------------------

    #[test]
    fn test_vector_clock_increment_and_merge() {
        let mut vc1 = AdvancedVectorClock::new();
        vc1.increment("A");
        vc1.increment("A");
        let mut vc2 = AdvancedVectorClock::new();
        vc2.increment("B");
        vc1.merge(&vc2);
        assert_eq!(vc1.get("A"), 2);
        assert_eq!(vc1.get("B"), 1);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let mut vc1 = AdvancedVectorClock::new();
        vc1.increment("A");
        let mut vc2 = AdvancedVectorClock::new();
        vc2.increment("A");
        vc2.increment("B");
        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut vc1 = AdvancedVectorClock::new();
        vc1.increment("A");
        let mut vc2 = AdvancedVectorClock::new();
        vc2.increment("B");
        assert!(vc1.concurrent_with(&vc2));
    }

    // -- G-Counter ----------------------------------------------------------

    #[test]
    fn test_g_counter_merge() {
        let mut c1 = GCounter::new();
        c1.increment("A", 3);
        c1.increment("B", 5);
        let mut c2 = GCounter::new();
        c2.increment("A", 7);
        c2.increment("C", 2);
        c1.merge(&c2);
        assert_eq!(c1.value(), 7 + 5 + 2);
    }

    // -- PN-Counter ---------------------------------------------------------

    #[test]
    fn test_pn_counter_increments_decrements() {
        let mut c = PNCounter::new();
        c.increment("A", 10);
        c.increment("B", -3);
        c.increment("A", -2);
        assert_eq!(c.value(), 5);
    }

    #[test]
    fn test_pn_counter_merge() {
        let mut c1 = PNCounter::new();
        c1.increment("A", 5);
        let mut c2 = PNCounter::new();
        c2.increment("A", -3);
        c1.merge(&c2);
        assert_eq!(c1.value(), 2);
    }

    // -- OR-Set -------------------------------------------------------------

    #[test]
    fn test_or_set_add_remove() {
        let mut s = ORSet::new();
        s.add("x", "t1");
        s.add("y", "t2");
        assert!(s.contains("x"));
        assert_eq!(s.len(), 2);
        s.remove("x");
        assert!(!s.contains("x"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_or_set_merge_concurrent_add() {
        let mut s1 = ORSet::new();
        s1.add("x", "t1");
        let mut s2 = ORSet::new();
        s2.add("x", "t2");
        s1.merge(&s2);
        assert!(s1.contains("x"));
        assert_eq!(s1.len(), 1);
        // Both tags survive -- element still present.
    }

    #[test]
    fn test_or_set_merge_remove_survives_concurrent_add() {
        let mut s1 = ORSet::new();
        s1.add("x", "t1");
        let mut s2 = ORSet::new();
        s2.add("x", "t2");
        s2.remove("x");
        s1.merge(&s2);
        // t1 was not in s2's tombstones, so "x" survives.
        assert!(s1.contains("x"));
    }

    // -- Raft simulation ----------------------------------------------------

    #[test]
    fn test_raft_election() {
        let mut node = RaftSimNode::new("N1", vec!["N2".into(), "N3".into()]);
        node.start_election();
        assert_eq!(node.role, RaftRole::Candidate);
        assert_eq!(node.term, 1);
        node.record_vote("N2");
        assert!(node.has_majority());
        node.become_leader();
        assert_eq!(node.role, RaftRole::Leader);
    }

    #[test]
    fn test_raft_log_replication() {
        let mut leader = RaftSimNode::new("L", vec!["F1".into(), "F2".into()]);
        leader.role = RaftRole::Leader;
        leader.term = 1;
        leader.append_entry("SET a=1");
        leader.append_entry("SET b=2");
        leader.commit();
        let applied = leader.apply_log();
        assert_eq!(applied, vec!["SET a=1", "SET b=2"]);
    }

    // -- DHT ---------------------------------------------------------------

    #[test]
    fn test_dht_put_get() {
        let mut dht = DistributedHashTable::new(1);
        dht.add_node("N1");
        dht.add_node("N2");
        dht.add_node("N3");
        dht.put("foo", "bar");
        assert_eq!(dht.get("foo").unwrap(), "bar");
    }

    #[test]
    fn test_dht_delete() {
        let mut dht = DistributedHashTable::new(1);
        dht.add_node("N1");
        dht.put("k", "v");
        assert!(dht.delete("k"));
        assert!(dht.get("k").is_none());
    }

    #[test]
    fn test_dht_replication() {
        let mut dht = DistributedHashTable::new(2);
        dht.add_node("A");
        dht.add_node("B");
        dht.add_node("C");
        let nodes = dht.put("key", "val");
        assert!(nodes.len() <= 2);
        // Should be retrievable regardless of which replica is queried.
        assert_eq!(dht.get("key").unwrap(), "val");
    }
}
