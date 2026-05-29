/// Advanced database internals: B+ tree, query planner, join algorithms,
/// ACID transactions, and buffer pool manager.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// B+ Tree Index
// ---------------------------------------------------------------------------

const ORDER: usize = 4;

#[derive(Debug, Clone)]
struct BNode<K: Ord + Clone, V: Clone> {
    keys: Vec<K>,
    vals: Vec<V>,
    children: Vec<BNode<K, V>>,
    is_leaf: bool,
}

#[derive(Debug)]
pub struct BPlusTree<K: Ord + Clone, V: Clone> {
    root: BNode<K, V>,
    len: usize,
}

impl<K: Ord + Clone + std::fmt::Debug, V: Clone + std::fmt::Debug> BPlusTree<K, V> {
    pub fn new() -> Self {
        Self {
            root: BNode { keys: vec![], vals: vec![], children: vec![], is_leaf: true },
            len: 0,
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let split = Self::ins(&mut self.root, key, value);
        if let Some((sep, new)) = split {
            let old = std::mem::replace(&mut self.root, BNode {
                keys: vec![sep], vals: vec![], children: vec![], is_leaf: false,
            });
            self.root.children = vec![old, new];
        }
        self.len += 1;
    }

    fn ins(node: &mut BNode<K, V>, key: K, val: V) -> Option<(K, BNode<K, V>)> {
        if node.is_leaf {
            let pos = node.keys.binary_search(&key).unwrap_or_else(|e| e);
            node.keys.insert(pos, key);
            node.vals.insert(pos, val);
            return if node.keys.len() >= ORDER { Self::split_leaf(node) } else { None };
        }
        let idx = node.keys.binary_search(&key).map(|i| i + 1).unwrap_or_else(|e| e);
        let split = Self::ins(&mut node.children[idx], key, val);
        if let Some((sep, new)) = split {
            let pos = node.keys.binary_search(&sep).unwrap_or_else(|e| e);
            node.keys.insert(pos, sep);
            node.children.insert(pos + 1, new);
            return if node.keys.len() >= ORDER { Self::split_internal(node) } else { None };
        }
        None
    }

    fn split_leaf(n: &mut BNode<K, V>) -> Option<(K, BNode<K, V>)> {
        let mid = n.keys.len() / 2;
        let sep = n.keys[mid].clone();
        Some((sep, BNode {
            keys: n.keys.split_off(mid),
            vals: n.vals.split_off(mid),
            children: vec![],
            is_leaf: true,
        }))
    }

    fn split_internal(n: &mut BNode<K, V>) -> Option<(K, BNode<K, V>)> {
        let mid = n.keys.len() / 2;
        let sep = n.keys.remove(mid);
        Some((sep, BNode {
            keys: n.keys.split_off(mid),
            vals: vec![],
            children: n.children.split_off(mid + 1),
            is_leaf: false,
        }))
    }

    pub fn search(&self, key: &K) -> Option<&V> {
        Self::find(&self.root, key)
    }

    fn find<'a>(node: &'a BNode<K, V>, key: &K) -> Option<&'a V> {
        if node.is_leaf {
            return node.keys.binary_search(key).ok().and_then(|i| node.vals.get(i));
        }
        let idx = node.keys.binary_search(key).map(|i| i + 1).unwrap_or_else(|e| e);
        Self::find(&node.children[idx], key)
    }

    pub fn range_query(&self, lo: &K, hi: &K) -> Vec<&V> {
        let mut out = vec![];
        Self::range(&self.root, lo, hi, &mut out);
        out
    }

    fn range<'a>(n: &'a BNode<K, V>, lo: &K, hi: &K, out: &mut Vec<&'a V>) {
        if n.is_leaf {
            for (i, k) in n.keys.iter().enumerate() {
                if k >= lo && k <= hi { out.push(&n.vals[i]); }
            }
            return;
        }
        for (i, k) in n.keys.iter().enumerate() {
            if k >= lo { Self::range(&n.children[i], lo, hi, out); }
            if k > hi { return; }
        }
        Self::range(n.children.last().unwrap(), lo, hi, out);
    }

    pub fn len(&self) -> usize { self.len }
}

// ---------------------------------------------------------------------------
// Row / Table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum DataValue { Int(i64), Float(f64), Text(String), Null }

impl std::fmt::Display for DataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataValue::Int(i) => write!(f, "{}", i),
            DataValue::Float(v) => write!(f, "{}", v),
            DataValue::Text(s) => write!(f, "{}", s),
            DataValue::Null => write!(f, "NULL"),
        }
    }
}

impl DataValue {
    pub fn as_i64(&self) -> Option<i64> {
        if let DataValue::Int(i) = self { Some(*i) } else { None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Row { pub values: Vec<DataValue> }

#[derive(Debug, Clone)]
pub struct ColumnInfo { pub name: String }

#[derive(Debug, Clone)]
pub struct TableSchema { pub name: String, pub columns: Vec<ColumnInfo> }

#[derive(Debug, Clone)]
pub struct Table {
    pub schema: TableSchema,
    pub rows: Vec<Row>,
    pub indexes: HashMap<String, BPlusTree<i64, usize>>,
}

impl Table {
    pub fn new(name: &str, columns: Vec<&str>) -> Self {
        Self {
            schema: TableSchema {
                name: name.into(),
                columns: columns.iter().map(|c| ColumnInfo { name: c.into() }).collect(),
            },
            rows: vec![],
            indexes: HashMap::new(),
        }
    }

    pub fn create_index(&mut self, col: &str) {
        let ci = self.schema.columns.iter().position(|c| c.name == col).expect("no column");
        let mut tree = BPlusTree::new();
        for (ri, row) in self.rows.iter().enumerate() {
            if let Some(k) = row.values[ci].as_i64() { tree.insert(k, ri); }
        }
        self.indexes.insert(col.into(), tree);
    }

    pub fn insert_row(&mut self, row: Row) {
        let ri = self.rows.len();
        for (name, tree) in &mut self.indexes {
            let ci = self.schema.columns.iter().position(|c| c.name == *name).unwrap();
            if let Some(k) = row.values[ci].as_i64() { tree.insert(k, ri); }
        }
        self.rows.push(row);
    }
}

// ---------------------------------------------------------------------------
// Query Planner
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PlanNode {
    SeqScan { table: String, predicate: Option<Predicate> },
    IndexScan { table: String, column: String, key: i64 },
    HashJoin { left: Box<PlanNode>, right: Box<PlanNode>, lk: usize, rk: usize },
    MergeJoin { left: Box<PlanNode>, right: Box<PlanNode>, lk: usize, rk: usize },
    NestedLoop { left: Box<PlanNode>, right: Box<PlanNode>, lk: usize, rk: usize },
}

#[derive(Debug, Clone)]
pub struct Predicate { pub column: usize, pub op: CompareOp, pub value: DataValue }

#[derive(Debug, Clone)]
pub enum CompareOp { Eq, Gt, Lt }

pub struct QueryPlanner;

impl QueryPlanner {
    pub fn plan(
        tables: &HashMap<String, Table>,
        left_t: &str, right_t: &str, lk: usize, rk: usize,
        predicate: Option<Predicate>,
    ) -> PlanNode {
        let left_scan = PlanNode::SeqScan { table: left_t.into(), predicate };
        let right_scan = PlanNode::SeqScan { table: right_t.into(), predicate: None };

        let lsz = tables.get(left_t).map(|t| t.rows.len()).unwrap_or(0);
        let rsz = tables.get(right_t).map(|t| t.rows.len()).unwrap_or(0);

        if lsz > 50 || rsz > 50 {
            PlanNode::HashJoin { left: Box::new(left_scan), right: Box::new(right_scan), lk, rk }
        } else {
            PlanNode::NestedLoop { left: Box::new(left_scan), right: Box::new(right_scan), lk, rk }
        }
    }
}

// ---------------------------------------------------------------------------
// Join Algorithms
// ---------------------------------------------------------------------------

pub struct JoinExecutor;

impl JoinExecutor {
    /// Hash join -- builds on smaller side, probes with larger.
    pub fn hash_join(left: &[Row], right: &[Row], lk: usize, rk: usize) -> Vec<Row> {
        let (build, probe, bk, pk) = if left.len() <= right.len() {
            (left, right, lk, rk)
        } else {
            (right, left, rk, lk)
        };

        let mut ht: HashMap<String, Vec<&Row>> = HashMap::new();
        for r in build { ht.entry(r.values[bk].to_string()).or_default().push(r); }

        let mut out = vec![];
        for pr in probe {
            if let Some(m) = ht.get(&pr.values[pk].to_string()) {
                for br in m {
                    let mut v = pr.values.clone();
                    v.extend(br.values.clone());
                    out.push(Row { values: v });
                }
            }
        }
        out
    }

    /// Merge join -- both inputs must be sorted on the join key.
    pub fn merge_join(left: &[Row], right: &[Row], lk: usize, rk: usize) -> Vec<Row> {
        let mut out = vec![];
        let (mut li, mut ri) = (0, 0);
        while li < left.len() && ri < right.len() {
            match Self::cmp(&left[li].values[lk], &right[ri].values[rk]) {
                std::cmp::Ordering::Less => li += 1,
                std::cmp::Ordering::Greater => ri += 1,
                std::cmp::Ordering::Equal => {
                    let ri_save = ri;
                    while li < left.len() && Self::cmp(&left[li].values[lk], &right[ri_save].values[rk]).is_eq() {
                        ri = ri_save;
                        while ri < right.len() && Self::cmp(&right[ri].values[rk], &left[li].values[lk]).is_eq() {
                            let mut v = left[li].values.clone();
                            v.extend(right[ri].values.clone());
                            out.push(Row { values: v });
                            ri += 1;
                        }
                        li += 1;
                    }
                }
            }
        }
        out
    }

    fn cmp(a: &DataValue, b: &DataValue) -> std::cmp::Ordering {
        match (a, b) {
            (DataValue::Int(x), DataValue::Int(y)) => x.cmp(y),
            (DataValue::Float(x), DataValue::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
            (DataValue::Text(x), DataValue::Text(y)) => x.cmp(y),
            (DataValue::Null, DataValue::Null) => std::cmp::Ordering::Equal,
            (DataValue::Null, _) => std::cmp::Ordering::Less,
            (_, DataValue::Null) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction Management (ACID)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState { Active, Committed, Aborted }

#[derive(Debug, Clone)]
pub struct WriteRecord { pub table: String, pub row_id: usize, pub old: Row, pub new: Row }

#[derive(Debug, Clone, Copy)]
pub enum IsolationLevel { ReadUncommitted, ReadCommitted, RepeatableRead, Serializable }

#[derive(Debug)]
pub struct Transaction {
    pub id: u64,
    pub state: TxnState,
    pub writes: Vec<WriteRecord>,
    pub isolation: IsolationLevel,
}

#[derive(Debug)]
pub enum TxnError { NotFound, NotActive }

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TxnError::NotFound => write!(f, "transaction not found"),
            TxnError::NotActive => write!(f, "transaction not active"),
        }
    }
}

pub struct TransactionManager {
    next_id: u64,
    active: HashMap<u64, Transaction>,
}

impl TransactionManager {
    pub fn new() -> Self { Self { next_id: 1, active: HashMap::new() } }

    pub fn begin(&mut self, iso: IsolationLevel) -> u64 {
        let id = self.next_id; self.next_id += 1;
        self.active.insert(id, Transaction { id, state: TxnState::Active, writes: vec![], isolation: iso });
        id
    }

    pub fn record_write(&mut self, id: u64, rec: WriteRecord) -> Result<(), TxnError> {
        let t = self.active.get_mut(&id).ok_or(TxnError::NotFound)?;
        if t.state != TxnState::Active { return Err(TxnError::NotActive); }
        t.writes.push(rec);
        Ok(())
    }

    pub fn commit(&mut self, id: u64) -> Result<Vec<WriteRecord>, TxnError> {
        let t = self.active.get_mut(&id).ok_or(TxnError::NotFound)?;
        if t.state != TxnState::Active { return Err(TxnError::NotActive); }
        t.state = TxnState::Committed;
        let w = t.writes.clone();
        self.active.remove(&id);
        Ok(w)
    }

    pub fn abort(&mut self, id: u64) -> Result<Vec<WriteRecord>, TxnError> {
        let t = self.active.get_mut(&id).ok_or(TxnError::NotFound)?;
        if t.state != TxnState::Active { return Err(TxnError::NotActive); }
        let w = t.writes.clone();
        self.active.remove(&id);
        Ok(w)
    }

    pub fn active_count(&self) -> usize { self.active.len() }
}

// ---------------------------------------------------------------------------
// Buffer Pool Manager
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Page { pub id: usize, pub data: Vec<u8>, pub dirty: bool, pub pin: u32 }

pub struct BufferPool {
    cap: usize,
    pages: HashMap<usize, Page>,
    lru: VecDeque<usize>,
    reads: u64,
    writes: u64,
}

impl BufferPool {
    pub fn new(cap: usize) -> Self {
        Self { cap, pages: HashMap::new(), lru: VecDeque::new(), reads: 0, writes: 0 }
    }

    pub fn fetch(&mut self, id: usize) -> &mut Page {
        if let Some(p) = self.pages.get_mut(&id) {
            p.pin += 1;
            self.lru.retain(|&i| i != id);
            self.lru.push_front(id);
            return self.pages.get_mut(&id).unwrap();
        }
        if self.pages.len() >= self.cap { self.evict(); }
        self.reads += 1;
        self.pages.insert(id, Page { id, data: vec![0u8; 4096], dirty: false, pin: 1 });
        self.lru.push_front(id);
        self.pages.get_mut(&id).unwrap()
    }

    pub fn unpin(&mut self, id: usize, dirty: bool) {
        if let Some(p) = self.pages.get_mut(&id) {
            if dirty { p.dirty = true; }
            if p.pin > 0 { p.pin -= 1; }
        }
    }

    pub fn flush(&mut self, id: usize) {
        if let Some(p) = self.pages.get_mut(&id) {
            if p.dirty { self.writes += 1; p.dirty = false; }
        }
    }

    fn evict(&mut self) {
        while let Some(id) = self.lru.pop_back() {
            if let Some(p) = self.pages.get(&id) {
                if p.pin == 0 {
                    if p.dirty { self.writes += 1; }
                    self.pages.remove(&id);
                    return;
                }
                self.lru.push_front(id);
            }
        }
    }

    pub fn stats(&self) -> (u64, u64) { (self.reads, self.writes) }
    pub fn used(&self) -> usize { self.pages.len() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btree_insert_search() {
        let mut t = BPlusTree::new();
        t.insert(10, "a"); t.insert(5, "b"); t.insert(15, "c");
        assert_eq!(t.search(&10), Some(&"a"));
        assert_eq!(t.search(&5), Some(&"b"));
        assert_eq!(t.search(&99), None);
    }

    #[test]
    fn btree_split_many() {
        let mut t = BPlusTree::new();
        for i in 0..20 { t.insert(i, i * 10); }
        assert_eq!(t.len(), 20);
        assert_eq!(t.search(&19), Some(&190));
    }

    #[test]
    fn btree_range() {
        let mut t = BPlusTree::new();
        for i in 0..10 { t.insert(i, i); }
        let r = t.range_query(&3, &7);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn table_index_roundtrip() {
        let mut tbl = Table::new("users", vec!["id", "name"]);
        tbl.insert_row(Row { values: vec![DataValue::Int(1), DataValue::Text("A".into())] });
        tbl.insert_row(Row { values: vec![DataValue::Int(2), DataValue::Text("B".into())] });
        tbl.create_index("id");
        assert_eq!(tbl.rows.len(), 2);
    }

    #[test]
    fn hash_join_basic() {
        let l = vec![Row { values: vec![DataValue::Int(1), DataValue::Text("a".into())] }];
        let r = vec![Row { values: vec![DataValue::Int(1), DataValue::Text("x".into())] },
                     Row { values: vec![DataValue::Int(2), DataValue::Text("y".into())] }];
        let j = JoinExecutor::hash_join(&l, &r, 0, 0);
        assert_eq!(j.len(), 1);
        assert_eq!(j[0].values[0], DataValue::Int(1));
    }

    #[test]
    fn hash_join_many_to_many() {
        let l = vec![
            Row { values: vec![DataValue::Int(1)] },
            Row { values: vec![DataValue::Int(1)] },
        ];
        let r = vec![
            Row { values: vec![DataValue::Int(1)] },
            Row { values: vec![DataValue::Int(1)] },
        ];
        assert_eq!(JoinExecutor::hash_join(&l, &r, 0, 0).len(), 4);
    }

    #[test]
    fn merge_join_sorted() {
        let l = vec![
            Row { values: vec![DataValue::Int(1)] },
            Row { values: vec![DataValue::Int(2)] },
            Row { values: vec![DataValue::Int(3)] },
        ];
        let r = vec![
            Row { values: vec![DataValue::Int(1)] },
            Row { values: vec![DataValue::Int(3)] },
        ];
        let j = JoinExecutor::merge_join(&l, &r, 0, 0);
        assert_eq!(j.len(), 2);
    }

    #[test]
    fn txn_commit() {
        let mut tm = TransactionManager::new();
        let id = tm.begin(IsolationLevel::ReadCommitted);
        tm.record_write(id, WriteRecord {
            table: "t".into(), row_id: 0,
            old: Row { values: vec![DataValue::Int(1)] },
            new: Row { values: vec![DataValue::Int(2)] },
        }).unwrap();
        let w = tm.commit(id).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn txn_abort() {
        let mut tm = TransactionManager::new();
        let id = tm.begin(IsolationLevel::Serializable);
        tm.record_write(id, WriteRecord {
            table: "t".into(), row_id: 0,
            old: Row { values: vec![DataValue::Int(10)] },
            new: Row { values: vec![DataValue::Int(20)] },
        }).unwrap();
        let w = tm.abort(id).unwrap();
        assert_eq!(w[0].old.values[0], DataValue::Int(10));
    }

    #[test]
    fn txn_double_commit_err() {
        let mut tm = TransactionManager::new();
        let id = tm.begin(IsolationLevel::ReadUncommitted);
        tm.commit(id).unwrap();
        assert!(tm.commit(id).is_err());
    }

    #[test]
    fn bufpool_evict_lru() {
        let mut bp = BufferPool::new(3);
        bp.fetch(0); bp.fetch(1); bp.fetch(2);
        bp.unpin(0, false); bp.unpin(1, false); bp.unpin(2, false);
        bp.fetch(3);
        assert_eq!(bp.used(), 3);
    }

    #[test]
    fn bufpool_dirty_flush() {
        let mut bp = BufferPool::new(2);
        let p = bp.fetch(0);
        p.data[0] = 42;
        bp.unpin(0, true);
        bp.flush(0);
        let (r, w) = bp.stats();
        assert_eq!(r, 1);
        assert_eq!(w, 1);
    }

    #[test]
    fn planner_large_uses_hash() {
        let mut tables = HashMap::new();
        let mut t = Table::new("o", vec!["id", "uid"]);
        for i in 0..100 { t.insert_row(Row { values: vec![DataValue::Int(i), DataValue::Int(i % 10)] }); }
        tables.insert("o".into(), t);
        tables.insert("u".into(), Table::new("u", vec!["id"]));
        let p = QueryPlanner::plan(&tables, "o", "u", 1, 0, None);
        assert!(matches!(p, PlanNode::HashJoin { .. }));
    }

    #[test]
    fn planner_small_uses_nested() {
        let mut tables = HashMap::new();
        tables.insert("a".into(), Table::new("a", vec!["id"]));
        tables.insert("b".into(), Table::new("b", vec!["id"]));
        let p = QueryPlanner::plan(&tables, "a", "b", 0, 0, None);
        assert!(matches!(p, PlanNode::NestedLoop { .. }));
    }
}
