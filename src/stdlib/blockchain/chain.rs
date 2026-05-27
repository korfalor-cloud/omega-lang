/// Blockchain implementation with blocks, transactions, Merkle trees, and mining.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Block {
    pub index: u64,
    pub timestamp: u64,
    pub previous_hash: String,
    pub hash: String,
    pub nonce: u64,
    pub transactions: Vec<Transaction>,
    pub merkle_root: String,
}

#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: String,
    pub sender: String,
    pub receiver: String,
    pub amount: f64,
    pub fee: f64,
    pub timestamp: u64,
    pub signature: String,
}

impl Transaction {
    pub fn new(sender: &str, receiver: &str, amount: f64, fee: f64) -> Self {
        let timestamp = current_timestamp();
        let id = compute_hash(&format!("{}{}{}{}{}", sender, receiver, amount, fee, timestamp));
        Self {
            id,
            sender: sender.to_string(),
            receiver: receiver.to_string(),
            amount,
            fee,
            timestamp,
            signature: String::new(),
        }
    }

    pub fn sign(&mut self, private_key: &str) {
        self.signature = compute_hash(&format!("{}{}", self.id, private_key));
    }

    pub fn data_string(&self) -> String {
        format!("{}{}{}{}{}", self.sender, self.receiver, self.amount, self.fee, self.timestamp)
    }
}

#[derive(Debug, Clone)]
pub struct Blockchain {
    pub chain: Vec<Block>,
    pub pending_transactions: Vec<Transaction>,
    pub difficulty: usize,
    pub balances: HashMap<String, f64>,
    pub mining_reward: f64,
    block_size: usize,
}

impl Blockchain {
    pub fn new(difficulty: usize) -> Self {
        let mut chain = Self {
            chain: Vec::new(),
            pending_transactions: Vec::new(),
            difficulty,
            balances: HashMap::new(),
            mining_reward: 50.0,
            block_size: 10,
        };
        chain.create_genesis_block();
        chain
    }

    fn create_genesis_block(&mut self) {
        let genesis = Block {
            index: 0,
            timestamp: 0,
            previous_hash: "0".repeat(64),
            hash: String::new(),
            nonce: 0,
            transactions: Vec::new(),
            merkle_root: String::new(),
        };
        let mut genesis = genesis;
        genesis.merkle_root = compute_merkle_root(&genesis.transactions);
        genesis.hash = self.calculate_block_hash(&genesis);
        self.chain.push(genesis);
    }

    pub fn add_transaction(&mut self, transaction: Transaction) -> Result<(), String> {
        if transaction.sender != "SYSTEM" {
            let balance = self.balances.get(&transaction.sender).copied().unwrap_or(0.0);
            if balance < transaction.amount + transaction.fee {
                return Err("Insufficient balance".to_string());
            }
        }
        self.pending_transactions.push(transaction);
        Ok(())
    }

    pub fn mine_pending_transactions(&mut self, miner_address: &str) -> Result<(), String> {
        if self.pending_transactions.is_empty() {
            return Err("No pending transactions".to_string());
        }

        // Take up to block_size transactions
        let transactions: Vec<Transaction> = self.pending_transactions
            .drain(..self.pending_transactions.len().min(self.block_size))
            .collect();

        // Add mining reward
        let reward_tx = Transaction::new("SYSTEM", miner_address, self.mining_reward, 0.0);
        let mut block_transactions = vec![reward_tx];
        block_transactions.extend(transactions);

        let previous_hash = self.chain.last().unwrap().hash.clone();
        let merkle_root = compute_merkle_root(&block_transactions);

        let mut block = Block {
            index: self.chain.len() as u64,
            timestamp: current_timestamp(),
            previous_hash,
            hash: String::new(),
            nonce: 0,
            transactions: block_transactions,
            merkle_root,
        };

        // Proof of work
        let target = "0".repeat(self.difficulty);
        loop {
            block.hash = self.calculate_block_hash(&block);
            if block.hash.starts_with(&target) {
                break;
            }
            block.nonce += 1;
        }

        // Update balances
        for tx in &block.transactions {
            if tx.sender != "SYSTEM" {
                let sender_balance = self.balances.entry(tx.sender.clone()).or_insert(0.0);
                *sender_balance -= tx.amount + tx.fee;
            }
            let receiver_balance = self.balances.entry(tx.receiver.clone()).or_insert(0.0);
            *receiver_balance += tx.amount;
        }

        self.chain.push(block);
        Ok(())
    }

    fn calculate_block_hash(&self, block: &Block) -> String {
        let data = format!("{}{}{}{}{}",
            block.index, block.timestamp, block.previous_hash, block.merkle_root, block.nonce);
        compute_hash(&data)
    }

    pub fn is_valid(&self) -> bool {
        for i in 1..self.chain.len() {
            let current = &self.chain[i];
            let previous = &self.chain[i - 1];

            // Verify hash
            if current.hash != self.calculate_block_hash(current) {
                return false;
            }

            // Verify chain link
            if current.previous_hash != previous.hash {
                return false;
            }

            // Verify proof of work
            let target = "0".repeat(self.difficulty);
            if !current.hash.starts_with(&target) {
                return false;
            }

            // Verify merkle root
            if current.merkle_root != compute_merkle_root(&current.transactions) {
                return false;
            }
        }
        true
    }

    pub fn get_balance(&self, address: &str) -> f64 {
        self.balances.get(address).copied().unwrap_or(0.0)
    }

    pub fn block_count(&self) -> usize {
        self.chain.len()
    }

    pub fn total_transactions(&self) -> usize {
        self.chain.iter().map(|b| b.transactions.len()).sum()
    }

    pub fn get_block(&self, index: u64) -> Option<&Block> {
        self.chain.get(index as usize)
    }

    pub fn get_transaction(&self, tx_id: &str) -> Option<&Transaction> {
        for block in &self.chain {
            for tx in &block.transactions {
                if tx.id == tx_id {
                    return Some(tx);
                }
            }
        }
        None
    }

    pub fn pending_count(&self) -> usize {
        self.pending_transactions.len()
    }
}

/// Compute Merkle root of transactions.
pub fn compute_merkle_root(transactions: &[Transaction]) -> String {
    if transactions.is_empty() {
        return compute_hash("");
    }

    let mut hashes: Vec<String> = transactions.iter().map(|tx| {
        compute_hash(&tx.data_string())
    }).collect();

    while hashes.len() > 1 {
        let mut next_level = Vec::new();
        for chunk in hashes.chunks(2) {
            let combined = if chunk.len() == 2 {
                format!("{}{}", chunk[0], chunk[1])
            } else {
                format!("{}{}", chunk[0], chunk[0])
            };
            next_level.push(compute_hash(&combined));
        }
        hashes = next_level;
    }

    hashes.into_iter().next().unwrap()
}

/// Simple hash function (SHA-256 placeholder).
pub fn compute_hash(data: &str) -> String {
    let bytes = data.as_bytes();
    let mut hash = [0u8; 32];

    for (i, &b) in bytes.iter().enumerate() {
        for j in 0..32 {
            hash[j] = hash[j].wrapping_add(
                b.wrapping_mul((i as u8).wrapping_add(j as u8))
                    .wrapping_add((j as u8).wrapping_mul(31))
            );
        }
    }

    // Double hash
    for (i, &b) in hash.iter().enumerate() {
        for j in 0..32 {
            hash[j] = hash[j].wrapping_add(
                b.wrapping_mul((i as u8).wrapping_add(j as u8))
                    .wrapping_mul(17)
            );
        }
    }

    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_blockchain() {
        let bc = Blockchain::new(2);
        assert_eq!(bc.block_count(), 1);
        assert!(bc.is_valid());
    }

    #[test]
    fn test_add_transaction() {
        let mut bc = Blockchain::new(1);
        bc.balances.insert("alice".to_string(), 100.0);
        let tx = Transaction::new("alice", "bob", 10.0, 0.01);
        assert!(bc.add_transaction(tx).is_ok());
    }

    #[test]
    fn test_insufficient_balance() {
        let mut bc = Blockchain::new(1);
        let tx = Transaction::new("alice", "bob", 100.0, 0.0);
        assert!(bc.add_transaction(tx).is_err());
    }

    #[test]
    fn test_mine_block() {
        let mut bc = Blockchain::new(1);
        bc.balances.insert("alice".to_string(), 100.0);
        let tx = Transaction::new("alice", "bob", 10.0, 0.01);
        bc.add_transaction(tx).unwrap();

        assert!(bc.mine_pending_transactions("miner").is_ok());
        assert_eq!(bc.block_count(), 2);
        assert!(bc.get_balance("bob") > 0.0);
        assert!(bc.is_valid());
    }

    #[test]
    fn test_merkle_root() {
        let txs = vec![
            Transaction::new("a", "b", 1.0, 0.0),
            Transaction::new("c", "d", 2.0, 0.0),
        ];
        let root = compute_merkle_root(&txs);
        assert!(!root.is_empty());
    }
}
