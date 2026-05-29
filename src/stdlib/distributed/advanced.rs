/// Advanced distributed systems: consistent hashing, vector clocks, CRDTs,
/// Raft consensus simulation, and a distributed hash table.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// -- Consistent hashing ----------------------------------------------------

const VIRTUAL_NODES: u64 = 128;

#[derive(Debug, Clone)]
pub struct ConsistentHashRing { ring: BTreeMap<u64, String>, nodes: HashSet<String> }

impl ConsistentHashRing {
    pub fn new() -> Self { Self { ring: BTreeMap::new(), nodes: HashSet::new() } }
    fn hash(key: &str) -> u64 { let mut h = DefaultHasher::new(); key.hash(&mut h); h.finish() }
    pub fn add_node(&mut self, node: &str) {
        self.nodes.insert(node.to_string());
        for i in 0..VIRTUAL_NODES { self.ring.insert(Self::hash(&format!("{}#{}", node, i)), node.into()); }
    }
    pub fn remove_node(&mut self, node: &str) {
        self.nodes.remove(node);
        for i in 0..VIRTUAL_NODES { self.ring.remove(&Self::hash(&format!("{}#{}", node, i))); }
    }
    pub fn get_node(&self, key: &str) -> Option<String> {
        if self.ring.is_empty() { return None; }
        let h = Self::hash(key);
        self.ring.range(h..).next().or_else(|| self.ring.iter().next()).map(|(_, v)| v.clone())
    }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

// -- Vector clocks ---------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvancedVectorClock { clocks: HashMap<String, u64> }

impl AdvancedVectorClock {
    pub fn new() -> Self { Self { clocks: HashMap::new() } }
    pub fn increment(&mut self, n: &str) { *self.clocks.entry(n.into()).or_insert(0) += 1; }
    pub fn merge(&mut self, other: &AdvancedVectorClock) {
        for (n, &t) in &other.clocks {
            let e = self.clocks.entry(n.clone()).or_insert(0);
            if t > *e { *e = t; }
        }
    }
    pub fn get(&self, n: &str) -> u64 { *self.clocks.get(n).unwrap_or(&0) }
    pub fn happens_before(&self, other: &AdvancedVectorClock) -> bool {
        let (mut all_leq, mut any_lt) = (true, false);
        for (n, &t) in &self.clocks {
            let ot = other.get(n);
            if t > ot { all_leq = false; break; }
            if t < ot { any_lt = true; }
        }
        all_leq && any_lt
    }
    pub fn concurrent_with(&self, other: &AdvancedVectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

// -- CRDTs -----------------------------------------------------------------

/// Grow-only counter.
#[derive(Debug, Clone)]
pub struct GCounter { counts: HashMap<String, u64> }
impl GCounter {
    pub fn new() -> Self { Self { counts: HashMap::new() } }
    pub fn increment(&mut self, n: &str, d: u64) { *self.counts.entry(n.into()).or_insert(0) += d; }
    pub fn value(&self) -> u64 { self.counts.values().sum() }
    pub fn merge(&mut self, o: &GCounter) {
        for (n, &c) in &o.counts {
            let e = self.counts.entry(n.clone()).or_insert(0);
            if c > *e { *e = c; }
        }
    }
}

/// Positive-Negative counter.
#[derive(Debug, Clone)]
pub struct PNCounter { pos: GCounter, neg: GCounter }
impl PNCounter {
    pub fn new() -> Self { Self { pos: GCounter::new(), neg: GCounter::new() } }
    pub fn increment(&mut self, n: &str, d: i64) {
        if d >= 0 { self.pos.increment(n, d as u64); } else { self.neg.increment(n, (-d) as u64); }
    }
    pub fn value(&self) -> i64 { self.pos.value() as i64 - self.neg.value() as i64 }
    pub fn merge(&mut self, o: &PNCounter) { self.pos.merge(&o.pos); self.neg.merge(&o.neg); }
}

/// Observed-Remove Set.
#[derive(Debug, Clone)]
pub struct ORSet { elems: HashMap<String, HashSet<String>>, tombs: HashMap<String, HashSet<String>> }
impl ORSet {
    pub fn new() -> Self { Self { elems: HashMap::new(), tombs: HashMap::new() } }
    pub fn add(&mut self, e: &str, tag: &str) { self.elems.entry(e.into()).or_default().insert(tag.into()); }
    pub fn remove(&mut self, e: &str) {
        if let Some(tags) = self.elems.remove(e) { self.tombs.entry(e.into()).or_default().extend(tags); }
    }
    pub fn contains(&self, e: &str) -> bool { self.elems.get(e).map_or(false, |t| !t.is_empty()) }
    pub fn members(&self) -> Vec<String> { self.elems.iter().filter(|(_, t)| !t.is_empty()).map(|(k, _)| k.clone()).collect() }
    pub fn merge(&mut self, o: &ORSet) {
        for (e, t) in &o.elems { self.elems.entry(e.clone()).or_default().extend(t.iter().cloned()); }
        for (e, t) in &o.tombs { self.tombs.entry(e.clone()).or_default().extend(t.iter().cloned()); }
        for (e, tomb) in &self.tombs { if let Some(tags) = self.elems.get_mut(e) { tags.retain(|t| !tomb.contains(t)); } }
    }
    pub fn len(&self) -> usize { self.elems.values().map(|t| t.len()).sum() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

// -- Raft consensus simulation ---------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RaftRole { Follower, Candidate, Leader }

#[derive(Debug, Clone)]
pub struct RaftSimNode {
    pub id: String, pub role: RaftRole, pub term: u64,
    pub log: Vec<(u64, String)>, pub commit_idx: usize,
    votes: HashSet<String>, pub peers: Vec<String>,
}
impl RaftSimNode {
    pub fn new(id: &str, peers: Vec<String>) -> Self {
        Self { id: id.into(), role: RaftRole::Follower, term: 0, log: vec![],
               commit_idx: 0, votes: HashSet::new(), peers }
    }
    pub fn start_election(&mut self) { self.role = RaftRole::Candidate; self.term += 1; self.votes.clear(); self.votes.insert(self.id.clone()); }
    pub fn request_vote(&mut self, _: &str, ct: u64) -> bool {
        if ct > self.term { self.term = ct; self.role = RaftRole::Follower; true } else { false }
    }
    pub fn record_vote(&mut self, v: &str) { self.votes.insert(v.into()); }
    pub fn has_majority(&self) -> bool { self.votes.len() > (self.peers.len() + 1) / 2 }
    pub fn become_leader(&mut self) { self.role = RaftRole::Leader; }
    pub fn append_entry(&mut self, cmd: &str) { self.log.push((self.term, cmd.into())); }
    pub fn commit(&mut self) { self.commit_idx = self.log.len(); }
    pub fn apply_log(&self) -> Vec<&str> { self.log[..self.commit_idx].iter().map(|(_, c)| c.as_str()).collect() }
}

// -- Distributed Hash Table ------------------------------------------------

#[derive(Debug, Clone)]
pub struct DistributedHashTable { ring: ConsistentHashRing, store: HashMap<String, String>, repl: usize }
impl DistributedHashTable {
    pub fn new(repl: usize) -> Self { Self { ring: ConsistentHashRing::new(), store: HashMap::new(), repl: repl.max(1) } }
    pub fn add_node(&mut self, n: &str) { self.ring.add_node(n); }
    pub fn put(&mut self, key: &str, val: &str) -> Vec<String> {
        let nodes = self.nodes_for(key);
        if let Some(p) = nodes.first() { self.store.insert(format!("{}@{}", p, key), val.into()); }
        nodes
    }
    pub fn get(&self, key: &str) -> Option<String> {
        for n in self.nodes_for(key) { if let Some(v) = self.store.get(&format!("{}@{}", n, key)) { return Some(v.clone()); } }
        None
    }
    pub fn delete(&mut self, key: &str) -> bool {
        let mut ok = false;
        for n in self.nodes_for(key) { if self.store.remove(&format!("{}@{}", n, key)).is_some() { ok = true; } }
        ok
    }
    fn nodes_for(&self, key: &str) -> Vec<String> {
        if self.ring.node_count() == 0 { return vec![]; }
        let h = ConsistentHashRing::hash(key);
        let (mut res, mut seen) = (vec![], HashSet::new());
        for (_, n) in self.ring.ring.range(h..) { if seen.insert(n.clone()) { res.push(n.clone()); } if res.len() >= self.repl { return res; } }
        for (_, n) in self.ring.ring.range(..h) { if seen.insert(n.clone()) { res.push(n.clone()); } if res.len() >= self.repl { return res; } }
        res
    }
    pub fn node_count(&self) -> usize { self.ring.node_count() }
}

// -- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_hash_ring_basic() {
        let mut r = ConsistentHashRing::new();
        r.add_node("A"); r.add_node("B"); r.add_node("C");
        assert_eq!(r.node_count(), 3);
        let n = r.get_node("some-key").unwrap();
        assert!(["A", "B", "C"].contains(&n.as_str()));
    }

    #[test]
    fn consistent_hash_ring_remove() {
        let mut r = ConsistentHashRing::new();
        r.add_node("A"); r.add_node("B"); r.remove_node("A");
        assert_eq!(r.node_count(), 1);
        assert_eq!(r.get_node("k").unwrap(), "B");
    }

    #[test]
    fn vector_clock_increment_merge() {
        let (mut v1, mut v2) = (AdvancedVectorClock::new(), AdvancedVectorClock::new());
        v1.increment("A"); v1.increment("A"); v2.increment("B");
        v1.merge(&v2);
        assert_eq!(v1.get("A"), 2);
        assert_eq!(v1.get("B"), 1);
    }

    #[test]
    fn vector_clock_happens_before() {
        let (mut v1, mut v2) = (AdvancedVectorClock::new(), AdvancedVectorClock::new());
        v1.increment("A"); v2.increment("A"); v2.increment("B");
        assert!(v1.happens_before(&v2));
        assert!(!v2.happens_before(&v1));
    }

    #[test]
    fn vector_clock_concurrent() {
        let (mut v1, mut v2) = (AdvancedVectorClock::new(), AdvancedVectorClock::new());
        v1.increment("A"); v2.increment("B");
        assert!(v1.concurrent_with(&v2));
    }

    #[test]
    fn g_counter_merge() {
        let (mut c1, mut c2) = (GCounter::new(), GCounter::new());
        c1.increment("A", 3); c1.increment("B", 5);
        c2.increment("A", 7); c2.increment("C", 2);
        c1.merge(&c2);
        assert_eq!(c1.value(), 14);
    }

    #[test]
    fn pn_counter_ops() {
        let mut c = PNCounter::new();
        c.increment("A", 10); c.increment("B", -3); c.increment("A", -2);
        assert_eq!(c.value(), 5);
    }

    #[test]
    fn pn_counter_merge() {
        let (mut c1, mut c2) = (PNCounter::new(), PNCounter::new());
        c1.increment("A", 5); c2.increment("A", -3);
        c1.merge(&c2);
        assert_eq!(c1.value(), 2);
    }

    #[test]
    fn or_set_add_remove() {
        let mut s = ORSet::new();
        s.add("x", "t1"); s.add("y", "t2");
        assert!(s.contains("x") && s.len() == 2);
        s.remove("x");
        assert!(!s.contains("x") && s.len() == 1);
    }

    #[test]
    fn or_set_concurrent_add_survives() {
        let (mut s1, mut s2) = (ORSet::new(), ORSet::new());
        s1.add("x", "t1"); s2.add("x", "t2");
        s1.merge(&s2);
        assert!(s1.contains("x"));
    }

    #[test]
    fn raft_election_and_log() {
        let mut n = RaftSimNode::new("N1", vec!["N2".into(), "N3".into()]);
        n.start_election();
        assert_eq!(n.role, RaftRole::Candidate);
        n.record_vote("N2");
        assert!(n.has_majority());
        n.become_leader();
        n.append_entry("SET a=1"); n.commit();
        assert_eq!(n.apply_log(), vec!["SET a=1"]);
    }

    #[test]
    fn dht_put_get_delete() {
        let mut dht = DistributedHashTable::new(1);
        dht.add_node("N1"); dht.add_node("N2"); dht.add_node("N3");
        dht.put("foo", "bar");
        assert_eq!(dht.get("foo").unwrap(), "bar");
        assert!(dht.delete("foo"));
        assert!(dht.get("foo").is_none());
    }

    #[test]
    fn dht_replication() {
        let mut dht = DistributedHashTable::new(2);
        dht.add_node("A"); dht.add_node("B"); dht.add_node("C");
        let nodes = dht.put("key", "val");
        assert!(nodes.len() <= 2);
        assert_eq!(dht.get("key").unwrap(), "val");
    }
}
