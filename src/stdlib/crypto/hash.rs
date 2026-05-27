use std::collections::HashMap;

pub struct OmegaHash;

impl OmegaHash {
    // FNV-1a hash
    pub fn fnv1a(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    // MurmurHash3 (32-bit)
    pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
        const C1: u32 = 0xcc9e2d51;
        const C2: u32 = 0x1b873593;
        const R1: u32 = 15;
        const R2: u32 = 13;
        const M: u32 = 5;
        const N: u32 = 0xe6546b64;

        let mut hash = seed;
        let len = data.len() as u32;

        // Body
        let nblocks = len / 4;
        for i in 0..nblocks {
            let k = u32::from_le_bytes([
                data[(i * 4) as usize],
                data[(i * 4 + 1) as usize],
                data[(i * 4 + 2) as usize],
                data[(i * 4 + 3) as usize],
            ]);

            let k = k.wrapping_mul(C1);
            let k = k.rotate_left(R1);
            let k = k.wrapping_mul(C2);

            hash ^= k;
            hash = hash.rotate_left(R2);
            hash = hash.wrapping_mul(M).wrapping_add(N);
        }

        // Tail
        let tail = &data[(nblocks * 4) as usize..];
        let mut k: u32 = 0;
        match tail.len() {
            3 => {
                k ^= (tail[2] as u32) << 16;
                k ^= (tail[1] as u32) << 8;
                k ^= tail[0] as u32;
                k = k.wrapping_mul(C1);
                k = k.rotate_left(R1);
                k = k.wrapping_mul(C2);
                hash ^= k;
            }
            2 => {
                k ^= (tail[1] as u32) << 8;
                k ^= tail[0] as u32;
                k = k.wrapping_mul(C1);
                k = k.rotate_left(R1);
                k = k.wrapping_mul(C2);
                hash ^= k;
            }
            1 => {
                k ^= tail[0] as u32;
                k = k.wrapping_mul(C1);
                k = k.rotate_left(R1);
                k = k.wrapping_mul(C2);
                hash ^= k;
            }
            _ => {}
        }

        // Finalization
        hash ^= len;
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        hash
    }

    // DJB2 hash
    pub fn djb2(data: &[u8]) -> u64 {
        let mut hash: u64 = 5381;
        for &byte in data {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    // SDBM hash
    pub fn sdbm(data: &[u8]) -> u64 {
        let mut hash: u64 = 0;
        for &byte in data {
            hash = byte as u64
                .wrapping_add(hash << 6)
                .wrapping_add(hash << 16)
                .wrapping_sub(hash);
        }
        hash
    }

    // Adler-32 checksum
    pub fn adler32(data: &[u8]) -> u32 {
        const MOD: u32 = 65521;
        let mut a: u32 = 1;
        let mut b: u32 = 0;

        for &byte in data {
            a = (a + byte as u32) % MOD;
            b = (b + a) % MOD;
        }

        (b << 16) | a
    }

    // CRC-32
    pub fn crc32(data: &[u8]) -> u32 {
        const POLY: u32 = 0xEDB88320;
        let mut crc: u32 = 0xFFFFFFFF;

        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ POLY;
                } else {
                    crc >>= 1;
                }
            }
        }

        !crc
    }

    // CRC-16
    pub fn crc16(data: &[u8]) -> u16 {
        const POLY: u16 = 0xA001;
        let mut crc: u16 = 0xFFFF;

        for &byte in data {
            crc ^= byte as u16;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ POLY;
                } else {
                    crc >>= 1;
                }
            }
        }

        crc
    }

    // Simple hash table implementation
    pub fn hash_string(s: &str) -> usize {
        Self::fnv1a(s.as_bytes()) as usize
    }
}

pub struct HashMap_Omega<K, V> {
    buckets: Vec<Vec<(K, V)>>,
    size: usize,
    capacity: usize,
    load_factor: f64,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> HashMap_Omega<K, V> {
    pub fn new() -> Self {
        Self {
            buckets: vec![Vec::new(); 16],
            size: 0,
            capacity: 16,
            load_factor: 0.75,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buckets: vec![Vec::new(); capacity],
            size: 0,
            capacity,
            load_factor: 0.75,
        }
    }

    fn bucket_index(&self, key: &K) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.capacity
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if (self.size as f64 / self.capacity as f64) > self.load_factor {
            self.resize();
        }

        let index = self.bucket_index(&key);
        let bucket = &mut self.buckets[index];

        for (k, v) in bucket.iter_mut() {
            if *k == key {
                let old = v.clone();
                *v = value;
                return Some(old);
            }
        }

        bucket.push((key, value));
        self.size += 1;
        None
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.bucket_index(key);
        self.buckets[index]
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.bucket_index(key);
        let bucket = &mut self.buckets[index];
        if let Some(pos) = bucket.iter().position(|(k, _)| k == key) {
            self.size -= 1;
            Some(bucket.remove(pos).1)
        } else {
            None
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        let index = self.bucket_index(key);
        self.buckets[index].iter().any(|(k, _)| k == key)
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn resize(&mut self) {
        let new_capacity = self.capacity * 2;
        let mut new_buckets = vec![Vec::new(); new_capacity];

        for bucket in self.buckets.drain(..) {
            for (key, value) in bucket {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                key.hash(&mut hasher);
                let index = hasher.finish() as usize % new_capacity;
                new_buckets[index].push((key, value));
            }
        }

        self.buckets = new_buckets;
        self.capacity = new_capacity;
    }
}
