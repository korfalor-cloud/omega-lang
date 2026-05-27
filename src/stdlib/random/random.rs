use std::collections::HashMap;

pub struct OmegaRandom {
    state: u64,
}

impl OmegaRandom {
    pub fn new() -> Self {
        Self {
            state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    // Xorshift64 algorithm
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_int(&mut self) -> i64 {
        self.next_u64() as i64
    }

    pub fn next_int_range(&mut self, min: i64, max: i64) -> i64 {
        if min >= max {
            return min;
        }
        let range = (max - min) as u64;
        min + (self.next_u64() % range) as i64
    }

    pub fn next_float(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    pub fn next_float_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_float()
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_u64() % 2 == 0
    }

    pub fn next_bool_probability(&mut self, probability: f64) -> bool {
        self.next_float() < probability
    }

    pub fn next_char(&mut self) -> char {
        let index = self.next_int_range(0, 26);
        (b'a' + index as u8) as char
    }

    pub fn next_char_range(&mut self, min: char, max: char) -> char {
        let min_val = min as u32;
        let max_val = max as u32;
        let val = self.next_int_range(min_val as i64, max_val as i64 + 1);
        std::char::from_u32(val as u32).unwrap_or(min)
    }

    pub fn next_string(&mut self, length: usize) -> String {
        (0..length).map(|_| self.next_char()).collect()
    }

    pub fn next_alphanumeric(&mut self, length: usize) -> String {
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            .chars()
            .collect();
        (0..length)
            .map(|_| {
                let index = self.next_int_range(0, chars.len() as i64);
                chars[index as usize]
            })
            .collect()
    }

    pub fn choose<T: Clone>(&mut self, items: &[T]) -> Option<T> {
        if items.is_empty() {
            return None;
        }
        let index = self.next_int_range(0, items.len() as i64);
        Some(items[index as usize].clone())
    }

    pub fn shuffle<T: Clone>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_int_range(0, i as i64 + 1) as usize;
            items.swap(i, j);
        }
    }

    pub fn sample<T: Clone>(&mut self, items: &[T], count: usize) -> Vec<T> {
        let mut indices: Vec<usize> = (0..items.len()).collect();
        self.shuffle(&mut indices);
        indices
            .iter()
            .take(count)
            .map(|&i| items[i].clone())
            .collect()
    }

    pub fn weighted_choice<T: Clone>(&mut self, items: &[(T, f64)]) -> Option<T> {
        if items.is_empty() {
            return None;
        }

        let total_weight: f64 = items.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut random = self.next_float() * total_weight;
        for (item, weight) in items {
            random -= weight;
            if random <= 0.0 {
                return Some(item.clone());
            }
        }

        items.last().map(|(item, _)| item.clone())
    }

    // Distributions
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.next_float();
        let u2 = self.next_float();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }

    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.next_float().ln() / lambda
    }

    pub fn poisson(&mut self, lambda: f64) -> i64 {
        let l = (-lambda).exp();
        let mut k = 0;
        let mut p = 1.0;

        loop {
            k += 1;
            p *= self.next_float();
            if p < l {
                break;
            }
        }

        k - 1
    }

    pub fn binomial(&mut self, n: i64, p: f64) -> i64 {
        let mut successes = 0;
        for _ in 0..n {
            if self.next_bool_probability(p) {
                successes += 1;
            }
        }
        successes
    }

    pub fn geometric(&mut self, p: f64) -> i64 {
        let mut trials = 0;
        loop {
            trials += 1;
            if self.next_bool_probability(p) {
                break;
            }
        }
        trials
    }

    // Random bytes
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| (self.next_u64() % 256) as u8).collect()
    }

    // UUID v4 generation
    pub fn uuid_v4(&mut self) -> String {
        let bytes = self.bytes(16);
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            ((bytes[6] & 0x0F) | 0x40), bytes[7],
            ((bytes[8] & 0x3F) | 0x80), bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }

    // Random color generation
    pub fn hex_color(&mut self) -> String {
        format!("#{:06x}", self.next_u64() % 0xFFFFFF)
    }

    pub fn rgb_color(&mut self) -> (u8, u8, u8) {
        (
            (self.next_u64() % 256) as u8,
            (self.next_u64() % 256) as u8,
            (self.next_u64() % 256) as u8,
        )
    }

    pub fn hsl_color(&mut self) -> (f64, f64, f64) {
        (
            self.next_float_range(0.0, 360.0),
            self.next_float_range(0.0, 1.0),
            self.next_float_range(0.0, 1.0),
        )
    }

    // Random names (simple)
    pub fn first_name(&mut self) -> &str {
        let names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
            "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
            "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
            "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
        ];
        let index = self.next_int_range(0, names.len() as i64);
        names[index as usize]
    }

    pub fn last_name(&mut self) -> &str {
        let names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
            "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
            "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        ];
        let index = self.next_int_range(0, names.len() as i64);
        names[index as usize]
    }

    pub fn full_name(&mut self) -> String {
        format!("{} {}", self.first_name(), self.last_name())
    }

    // Random words
    pub fn word(&mut self) -> &str {
        let words = [
            "apple", "banana", "cherry", "dog", "elephant", "fish",
            "grape", "house", "igloo", "jungle", "kite", "lemon",
            "mango", "notebook", "orange", "piano", "queen", "rainbow",
            "sun", "tree", "umbrella", "violin", "water", "xylophone",
            "yellow", "zebra",
        ];
        let index = self.next_int_range(0, words.len() as i64);
        words[index as usize]
    }

    pub fn sentence(&mut self, word_count: usize) -> String {
        let mut sentence = String::new();
        for i in 0..word_count {
            if i > 0 {
                sentence.push(' ');
            }
            let word = self.word().to_string();
            if i == 0 {
                let first_char = word.chars().next().unwrap().to_uppercase().next().unwrap();
                sentence.push(first_char);
                sentence.push_str(&word[1..]);
            } else {
                sentence.push_str(&word);
            }
        }
        sentence.push('.');
        sentence
    }

    pub fn paragraph(&mut self, sentence_count: usize) -> String {
        (0..sentence_count)
            .map(|_| self.sentence(self.next_int_range(5, 15) as usize))
            .collect::<Vec<String>>()
            .join(" ")
    }
}

// Random distributions as standalone functions
pub fn random_normal(mean: f64, std_dev: f64, samples: usize) -> Vec<f64> {
    let mut rng = OmegaRandom::new();
    (0..samples).map(|_| rng.normal(mean, std_dev)).collect()
}

pub fn random_uniform(min: f64, max: f64, samples: usize) -> Vec<f64> {
    let mut rng = OmegaRandom::new();
    (0..samples).map(|_| rng.next_float_range(min, max)).collect()
}

pub fn random_integers(min: i64, max: i64, count: usize) -> Vec<i64> {
    let mut rng = OmegaRandom::new();
    (0..count).map(|_| rng.next_int_range(min, max)).collect()
}

// Fisher-Yates shuffle
pub fn random_shuffle<T: Clone>(items: &mut [T]) {
    let mut rng = OmegaRandom::new();
    rng.shuffle(items);
}

// Random subset
pub fn random_subset<T: Clone>(items: &[T], size: usize) -> Vec<T> {
    let mut rng = OmegaRandom::new();
    rng.sample(items, size)
}
