/// Data partitioning strategies for distributed systems.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    Hash,
    Range,
    RoundRobin,
    ConsistentHash { virtual_nodes: usize },
}

#[derive(Debug)]
pub struct Partitioner {
    strategy: PartitionStrategy,
    partitions: usize,
    ring: Vec<(u64, usize)>,
    counter: usize,
}

impl Partitioner {
    pub fn new(strategy: PartitionStrategy, partitions: usize) -> Self {
        let mut p = Self {
            strategy,
            partitions,
            ring: Vec::new(),
            counter: 0,
        };
        if matches!(p.strategy, PartitionStrategy::ConsistentHash { .. }) {
            p.build_ring();
        }
        p
    }

    fn build_ring(&mut self) {
        if let PartitionStrategy::ConsistentHash { virtual_nodes } = self.strategy {
            self.ring.clear();
            for partition in 0..self.partitions {
                for vnode in 0..virtual_nodes {
                    let key = format!("{}:{}", partition, vnode);
                    let hash = self.hash_key(&key);
                    self.ring.push((hash, partition));
                }
            }
            self.ring.sort_by_key(|&(hash, _)| hash);
        }
    }

    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn partition(&self, key: &str) -> usize {
        match &self.strategy {
            PartitionStrategy::Hash => {
                self.hash_key(key) as usize % self.partitions
            }
            PartitionStrategy::Range => {
                let first_char = key.chars().next().unwrap_or('a') as usize;
                first_char % self.partitions
            }
            PartitionStrategy::RoundRobin => {
                // This is mutable in practice, but for the interface we use hash
                self.hash_key(key) as usize % self.partitions
            }
            PartitionStrategy::ConsistentHash { .. } => {
                let hash = self.hash_key(key);
                match self.ring.binary_search_by_key(&hash, |&(h, _)| h) {
                    Ok(idx) => self.ring[idx].1,
                    Err(idx) => {
                        if idx >= self.ring.len() {
                            self.ring[0].1
                        } else {
                            self.ring[idx].1
                        }
                    }
                }
            }
        }
    }

    pub fn partition_with_round_robin(&mut self, _key: &str) -> usize {
        let partition = self.counter % self.partitions;
        self.counter += 1;
        partition
    }

    pub fn add_partition(&mut self) {
        self.partitions += 1;
        if matches!(self.strategy, PartitionStrategy::ConsistentHash { .. }) {
            self.build_ring();
        }
    }

    pub fn remove_partition(&mut self, partition: usize) {
        if partition < self.partitions {
            self.partitions -= 1;
            if matches!(self.strategy, PartitionStrategy::ConsistentHash { .. }) {
                self.build_ring();
            }
        }
    }

    pub fn partition_count(&self) -> usize {
        self.partitions
    }
}

/// Shard map for routing
#[derive(Debug)]
pub struct ShardMap {
    shards: Vec<Shard>,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub id: usize,
    pub start: u64,
    pub end: u64,
    pub node: String,
}

impl ShardMap {
    pub fn new() -> Self {
        Self { shards: Vec::new() }
    }

    pub fn add_shard(&mut self, start: u64, end: u64, node: &str) {
        let id = self.shards.len();
        self.shards.push(Shard {
            id,
            start,
            end,
            node: node.to_string(),
        });
        self.shards.sort_by_key(|s| s.start);
    }

    pub fn route(&self, key: u64) -> Option<&Shard> {
        self.shards.iter().find(|s| key >= s.start && key <= s.end)
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partition() {
        let partitioner = Partitioner::new(PartitionStrategy::Hash, 4);
        let p = partitioner.partition("test_key");
        assert!(p < 4);
    }

    #[test]
    fn test_consistent_hash() {
        let partitioner = Partitioner::new(
            PartitionStrategy::ConsistentHash { virtual_nodes: 100 },
            4,
        );
        let p = partitioner.partition("test_key");
        assert!(p < 4);
    }

    #[test]
    fn test_consistent_hash_stability() {
        let partitioner = Partitioner::new(
            PartitionStrategy::ConsistentHash { virtual_nodes: 100 },
            4,
        );
        let p1 = partitioner.partition("stable_key");
        let p2 = partitioner.partition("stable_key");
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_shard_map() {
        let mut shards = ShardMap::new();
        shards.add_shard(0, 99, "node1");
        shards.add_shard(100, 199, "node2");

        assert_eq!(shards.route(50).unwrap().node, "node1");
        assert_eq!(shards.route(150).unwrap().node, "node2");
    }
}
