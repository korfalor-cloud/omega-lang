//! Advanced probabilistic and balanced data structures.
//!
//! Provides: B-tree, Red-Black tree, Skip list, Bloom filter, and Count-Min sketch.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ─── B-Tree ──────────────────────────────────────────────────────────────────

const B: usize = 4;

pub struct OmegaBTree<K: Ord + Clone, V: Clone> {
    root: Option<Box<BNode<K, V>>>,
    len: usize,
}

struct BNode<K: Ord + Clone, V: Clone> {
    keys: Vec<K>,
    values: Vec<V>,
    children: Vec<Box<BNode<K, V>>>,
}

impl<K: Ord + Clone, V: Clone> BNode<K, V> {
    fn leaf() -> Self {
        Self { keys: Vec::new(), values: Vec::new(), children: Vec::new() }
    }
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl<K: Ord + Clone, V: Clone> OmegaBTree<K, V> {
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let root = self.root.get_or_insert_with(|| Box::new(BNode::leaf()));
        let old = Self::insert_into(root, key, value);
        if old.is_none() {
            self.len += 1;
        }
        if root.keys.len() >= 2 * B {
            let mut new_root = Box::new(BNode::leaf());
            let old_root = self.root.take().unwrap();
            let mid = old_root.keys.len() / 2;
            new_root.keys.push(old_root.keys[mid].clone());
            new_root.values.push(old_root.values[mid].clone());
            let (left, right) = Self::split_node(old_root, mid);
            new_root.children.push(left);
            new_root.children.push(right);
            self.root = Some(new_root);
        }
        old
    }

    fn insert_into(node: &mut Box<BNode<K, V>>, key: K, value: V) -> Option<V> {
        let pos = node.keys.binary_search(&key);
        match pos {
            Ok(i) => {
                let old = std::mem::replace(&mut node.values[i], value);
                Some(old)
            }
            Err(i) => {
                if node.is_leaf() {
                    node.keys.insert(i, key);
                    node.values.insert(i, value);
                    None
                } else {
                    let old = Self::insert_into(&mut node.children[i], key, value);
                    if node.children[i].keys.len() >= 2 * B {
                        let child = node.children.remove(i);
                        let mid = child.keys.len() / 2;
                        node.keys.insert(i, child.keys[mid].clone());
                        node.values.insert(i, child.values[mid].clone());
                        let (left, right) = Self::split_node(child, mid);
                        node.children.insert(i, right);
                        node.children.insert(i, left);
                    }
                    old
                }
            }
        }
    }

    fn split_node(
        mut node: Box<BNode<K, V>>,
        mid: usize,
    ) -> (Box<BNode<K, V>>, Box<BNode<K, V>>) {
        let right_keys = node.keys.split_off(mid + 1);
        let right_values = node.values.split_off(mid + 1);
        let right_children = if node.is_leaf() {
            Vec::new()
        } else {
            node.children.split_off(mid + 1)
        };
        node.keys.pop();
        node.values.pop();
        let right = Box::new(BNode { keys: right_keys, values: right_values, children: right_children });
        (node, right)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.root.as_ref().and_then(|n| Self::search(n, key))
    }

    fn search(node: &BNode<K, V>, key: &K) -> Option<&V> {
        match node.keys.binary_search(key) {
            Ok(i) => Some(&node.values[i]),
            Err(i) if !node.is_leaf() => Self::search(&node.children[i], key),
            _ => None,
        }
    }
}

// ─── Red-Black Tree ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Color { Red, Black }

struct RBNode<K: Ord, V> {
    key: K,
    value: V,
    color: Color,
    left: Option<Box<RBNode<K, V>>>,
    right: Option<Box<RBNode<K, V>>>,
}

pub struct RedBlackTree<K: Ord, V> {
    root: Option<Box<RBNode<K, V>>>,
    len: usize,
}

impl<K: Ord, V> RedBlackTree<K, V> {
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn is_red(n: &Option<Box<RBNode<K, V>>>) -> bool {
        n.as_ref().map_or(false, |x| x.color == Color::Red)
    }

    fn rotate_left(mut node: Box<RBNode<K, V>>) -> Box<RBNode<K, V>> {
        let mut x = node.right.take().unwrap();
        node.right = x.left.take();
        x.color = node.color;
        node.color = Color::Red;
        x.left = Some(node);
        x
    }

    fn rotate_right(mut node: Box<RBNode<K, V>>) -> Box<RBNode<K, V>> {
        let mut x = node.left.take().unwrap();
        node.left = x.right.take();
        x.color = node.color;
        node.color = Color::Red;
        x.right = Some(node);
        x
    }

    fn flip_colors(node: &mut Box<RBNode<K, V>>) {
        node.color = Color::Red;
        if let Some(ref mut l) = node.left { l.color = Color::Black; }
        if let Some(ref mut r) = node.right { r.color = Color::Black; }
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.root = Some(Self::bst_insert(self.root.take(), key, value));
        if let Some(ref mut r) = self.root { r.color = Color::Black; }
        self.len += 1;
    }

    fn bst_insert(node: Option<Box<RBNode<K, V>>>, key: K, value: V) -> Box<RBNode<K, V>> {
        let mut node = match node {
            None => return Box::new(RBNode { key, value, color: Color::Red, left: None, right: None }),
            Some(n) => n,
        };
        match key.cmp(&node.key) {
            std::cmp::Ordering::Less => node.left = Some(Self::bst_insert(node.left.take(), key, value)),
            std::cmp::Ordering::Greater => node.right = Some(Self::bst_insert(node.right.take(), key, value)),
            std::cmp::Ordering::Equal => { node.value = value; return node; }
        }
        if Self::is_red(&node.right) && !Self::is_red(&node.left) {
            node = Self::rotate_left(node);
        }
        if Self::is_red(&node.left) && node.left.as_ref().map_or(false, |l| Self::is_red(&l.left)) {
            node = Self::rotate_right(node);
        }
        if Self::is_red(&node.left) && Self::is_red(&node.right) {
            Self::flip_colors(&mut node);
        }
        node
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let mut cur = self.root.as_ref();
        while let Some(n) = cur {
            match key.cmp(&n.key) {
                std::cmp::Ordering::Less => cur = n.left.as_ref(),
                std::cmp::Ordering::Greater => cur = n.right.as_ref(),
                std::cmp::Ordering::Equal => return Some(&n.value),
            }
        }
        None
    }
}

// ─── Skip List (arena-based for safe Rust) ───────────────────────────────────

const MAX_LEVEL: usize = 16;

struct ArenaNode<K, V> {
    key: K,
    value: V,
    next: Vec<usize>, // indices into arena, usize::MAX = null
}

pub struct SkipList<K: Ord, V> {
    arena: Vec<ArenaNode<K, V>>,
    heads: Vec<usize>,
    level: usize,
    len: usize,
}

impl<K: Ord, V> SkipList<K, V> {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            heads: vec![usize::MAX; MAX_LEVEL],
            level: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn random_level() -> usize {
        let mut lvl = 0;
        while lvl < MAX_LEVEL - 1 && rand::random::<f64>() < 0.5 {
            lvl += 1;
        }
        lvl
    }

    pub fn insert(&mut self, key: K, value: V) {
        let new_level = Self::random_level();
        let new_idx = self.arena.len();

        let mut node = ArenaNode {
            key,
            value,
            next: vec![usize::MAX; new_level + 1],
        };

        for lvl in (0..=new_level.max(self.level)).rev() {
            let mut pred = usize::MAX;
            let mut cursor = self.heads[lvl];
            while cursor != usize::MAX && self.arena[cursor].key < node.key {
                pred = cursor;
                cursor = self.arena[cursor].next[lvl];
            }

            if lvl <= new_level {
                node.next[lvl] = cursor;
                if pred == usize::MAX {
                    self.heads[lvl] = new_idx;
                } else {
                    self.arena[pred].next[lvl] = new_idx;
                }
            }
        }

        self.arena.push(node);
        self.level = self.level.max(new_level);
        self.len += 1;
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        for lvl in (0..=self.level).rev() {
            let mut c = self.heads[lvl];
            while c != usize::MAX {
                match self.arena[c].key.cmp(key) {
                    std::cmp::Ordering::Less => c = self.arena[c].next[lvl],
                    std::cmp::Ordering::Equal => return Some(&self.arena[c].value),
                    std::cmp::Ordering::Greater => break,
                }
            }
        }
        None
    }
}

// ─── Bloom Filter ────────────────────────────────────────────────────────────

pub struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    len: usize,
}

impl BloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        Self { bits: vec![false; size], num_hashes, len: 0 }
    }

    pub fn with_fp_rate(expected_items: usize, fp_rate: f64) -> Self {
        let size = (-(expected_items as f64) * fp_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_hashes = ((size as f64 / expected_items as f64) * 2.0_f64.ln()).round() as usize;
        Self::new(size, num_hashes.max(1))
    }

    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut results = Vec::with_capacity(self.num_hashes);
        for i in 0..self.num_hashes {
            let mut h = DefaultHasher::new();
            item.hash(&mut h);
            i.hash(&mut h);
            results.push(h.finish() as usize % self.bits.len());
        }
        results
    }

    pub fn insert<T: Hash>(&mut self, item: &T) {
        for pos in self.hashes(item) {
            self.bits[pos] = true;
        }
        self.len += 1;
    }

    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        self.hashes(item).iter().all(|&pos| self.bits[pos])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.bits.fill(false);
        self.len = 0;
    }

    pub fn estimated_fpp(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count();
        let ratio = set_bits as f64 / self.bits.len() as f64;
        ratio.powi(self.num_hashes as i32)
    }
}

// ─── Count-Min Sketch ────────────────────────────────────────────────────────

pub struct CountMinSketch {
    table: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
    total: u64,
}

impl CountMinSketch {
    pub fn new(width: usize, depth: usize) -> Self {
        Self { table: vec![vec![0u64; width]; depth], width, depth, total: 0 }
    }

    pub fn with_error(epsilon: f64, delta: f64) -> Self {
        let width = (1.0 / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width, depth.max(1))
    }

    fn hash_at(item: &[u8], idx: usize, width: usize) -> usize {
        let mut h = DefaultHasher::new();
        item.hash(&mut h);
        idx.hash(&mut h);
        h.finish() as usize % width
    }

    fn to_bytes<T: Hash>(item: &T) -> Vec<u8> {
        let mut h = DefaultHasher::new();
        item.hash(&mut h);
        h.finish().to_le_bytes().to_vec()
    }

    pub fn increment<T: Hash>(&mut self, item: &T, count: u64) {
        let bytes = Self::to_bytes(item);
        for i in 0..self.depth {
            let pos = Self::hash_at(&bytes, i, self.width);
            self.table[i][pos] += count;
        }
        self.total += count;
    }

    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        let bytes = Self::to_bytes(item);
        (0..self.depth)
            .map(|i| self.table[i][Self::hash_at(&bytes, i, self.width)])
            .min()
            .unwrap_or(0)
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn clear(&mut self) {
        for row in &mut self.table {
            row.fill(0);
        }
        self.total = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- B-Tree --

    #[test]
    fn btree_insert_and_get() {
        let mut t = OmegaBTree::new();
        assert!(t.is_empty());
        for i in 0..20 {
            t.insert(i, i * 10);
        }
        assert_eq!(t.len(), 20);
        assert_eq!(t.get(&5), Some(&50));
        assert_eq!(t.get(&100), None);
    }

    #[test]
    fn btree_overwrite() {
        let mut t = OmegaBTree::new();
        t.insert(1, "a");
        t.insert(1, "b");
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&1), Some(&"b"));
    }

    #[test]
    fn btree_many_inserts() {
        let mut t = OmegaBTree::new();
        for i in 0..200 {
            t.insert(i, i);
        }
        assert_eq!(t.len(), 200);
        assert_eq!(t.get(&0), Some(&0));
        assert_eq!(t.get(&199), Some(&199));
    }

    // -- Red-Black Tree --

    #[test]
    fn rbtree_insert_and_get() {
        let mut t = RedBlackTree::new();
        for i in 0..50 {
            t.insert(i, format!("v{i}"));
        }
        assert_eq!(t.len(), 50);
        assert_eq!(t.get(&25), Some(&"v25".to_string()));
        assert_eq!(t.get(&100), None);
    }

    #[test]
    fn rbtree_reverse_insert() {
        let mut t = RedBlackTree::new();
        for i in (0..30).rev() {
            t.insert(i, i);
        }
        assert_eq!(t.len(), 30);
        assert_eq!(t.get(&0), Some(&0));
        assert_eq!(t.get(&29), Some(&29));
    }

    #[test]
    fn rbtree_overwrite() {
        let mut t = RedBlackTree::new();
        t.insert(1, 10);
        t.insert(1, 20);
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&1), Some(&20));
    }

    // -- Skip List --

    #[test]
    fn skiplist_insert_and_get() {
        let mut sl = SkipList::new();
        assert!(sl.is_empty());
        for i in 0..50 {
            sl.insert(i, i * 2);
        }
        assert_eq!(sl.len(), 50);
        assert_eq!(sl.get(&25), Some(&50));
        assert_eq!(sl.get(&99), None);
    }

    #[test]
    fn skiplist_reverse_insert() {
        let mut sl = SkipList::new();
        for i in (0..30).rev() {
            sl.insert(i, i);
        }
        assert_eq!(sl.len(), 30);
        assert_eq!(sl.get(&0), Some(&0));
        assert_eq!(sl.get(&29), Some(&29));
    }

    // -- Bloom Filter --

    #[test]
    fn bloom_basic() {
        let mut bf = BloomFilter::new(1000, 3);
        bf.insert(&"hello");
        bf.insert(&"world");
        assert!(bf.contains(&"hello"));
        assert!(bf.contains(&"world"));
        assert!(!bf.contains(&"missing"));
        assert_eq!(bf.len(), 2);
    }

    #[test]
    fn bloom_fp_rate_constructor() {
        let mut bf = BloomFilter::with_fp_rate(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&i);
        }
        assert_eq!(bf.len(), 1000);
        assert!(bf.contains(&500));
        assert!(bf.estimated_fpp() < 0.5);
    }

    #[test]
    fn bloom_clear() {
        let mut bf = BloomFilter::new(100, 3);
        bf.insert(&42);
        assert!(bf.contains(&42));
        bf.clear();
        assert!(!bf.contains(&42));
        assert!(bf.is_empty());
    }

    // -- Count-Min Sketch --

    #[test]
    fn cms_basic() {
        let mut cms = CountMinSketch::new(100, 5);
        cms.increment(&"foo", 10);
        cms.increment(&"bar", 20);
        assert!(cms.estimate(&"foo") >= 10);
        assert!(cms.estimate(&"bar") >= 20);
        assert_eq!(cms.total(), 30);
    }

    #[test]
    fn cms_heavy_hitters() {
        let mut cms = CountMinSketch::with_error(0.001, 0.01);
        for i in 0..100 {
            let count = if i < 5 { 1000 } else { 1 };
            for _ in 0..count {
                cms.increment(&i, 1);
            }
        }
        assert!(cms.estimate(&0) >= 1000);
        assert!(cms.estimate(&99) >= 1);
        assert_eq!(cms.total(), 5095);
    }

    #[test]
    fn cms_clear() {
        let mut cms = CountMinSketch::new(50, 3);
        cms.increment(&"test", 42);
        assert!(cms.estimate(&"test") >= 42);
        cms.clear();
        assert_eq!(cms.estimate(&"test"), 0);
        assert_eq!(cms.total(), 0);
    }
}
