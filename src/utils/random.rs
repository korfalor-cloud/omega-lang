use rand::Rng;

pub fn random_int(min: i64, max: i64) -> i64 {
    rand::thread_rng().gen_range(min..=max)
}

pub fn random_float(min: f64, max: f64) -> f64 {
    rand::thread_rng().gen_range(min..max)
}

pub fn random_bool() -> bool {
    rand::thread_rng().gen()
}

pub fn random_char() -> char {
    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".chars().collect();
    let index = rand::thread_rng().gen_range(0..chars.len());
    chars[index]
}

pub fn random_string(length: usize) -> String {
    (0..length).map(|_| random_char()).collect()
}

pub fn random_bytes(count: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; count];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

pub fn random_choice<T: Clone>(items: &[T]) -> Option<T> {
    if items.is_empty() {
        return None;
    }
    let index = rand::thread_rng().gen_range(0..items.len());
    Some(items[index].clone())
}

pub fn random_shuffle<T: Clone>(items: &[T]) -> Vec<T> {
    use rand::seq::SliceRandom;
    let mut result = items.to_vec();
    let mut rng = rand::thread_rng();
    result.shuffle(&mut rng);
    result
}

pub fn random_sample<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    use rand::seq::SliceRandom;
    let mut result = items.to_vec();
    let mut rng = rand::thread_rng();
    result.shuffle(&mut rng);
    result.truncate(n);
    result
}

pub fn random_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub fn random_hex(length: usize) -> String {
    let mut result = String::new();
    for _ in 0..length {
        let byte: u8 = rand::thread_rng().gen();
        result.push_str(&format!("{:02x}", byte));
    }
    result
}

pub fn random_alphanumeric(length: usize) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut result = String::new();
    for _ in 0..length {
        let index = rand::thread_rng().gen_range(0..CHARS.len());
        result.push(CHARS[index] as char);
    }
    result
}

pub fn random_normal(mean: f64, std_dev: f64) -> f64 {
    use rand_distr::Distribution;
    let normal = rand_distr::Normal::new(mean, std_dev).unwrap();
    normal.sample(&mut rand::thread_rng())
}

pub fn random_poisson(lambda: f64) -> u64 {
    use rand_distr::Distribution;
    let poisson = rand_distr::Poisson::new(lambda).unwrap();
    poisson.sample(&mut rand::thread_rng()) as u64
}
