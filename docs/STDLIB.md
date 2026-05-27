# Omega Standard Library

## Collections

### Vec<T>
Dynamic array implementation with rich methods.

```omega
let v = vec![1, 2, 3, 4, 5]
v.push(6)
v.pop()
v.map(|x| x * 2)
v.filter(|x| x > 3)
v.fold(0, |acc, x| acc + x)
v.sort()
v.reverse()
v.binary_search(3)
v.windows(3)
v.chunks(2)
v.group_by(|x| x % 2)
v.unique()
v.shuffle()
v.sample(3)
```

### Map<K, V>
Hash map preserving insertion order.

```omega
let m = {"key": "value", "count": 42}
m.insert("new_key", "new_value")
m.get("key")
m.remove("key")
m.has("key")
m.keys()
m.values()
m.entries()
m.merge(other_map)
m.filter(|k, v| v > 10)
m.sort_by_key(|k, v| k)
```

### Set<T>
Hash set implementation.

```omega
let s = set![1, 2, 3, 4, 5]
s.add(6)
s.remove(1)
s.has(2)
s.union(other_set)
s.intersection(other_set)
s.difference(other_set)
s.symmetric_difference(other_set)
s.is_subset(other_set)
s.is_superset(other_set)
```

### Deque<T>
Double-ended queue.

```omega
let dq = Deque::new()
dq.push_front(1)
dq.push_back(2)
dq.pop_front()
dq.pop_back()
```

### Heap<T>
Binary heap (priority queue).

```omega
let h = Heap::new()
h.push(3)
h.push(1)
h.push(2)
h.peek()  // Returns 1 (min)
h.pop()   // Returns 1
```

## I/O

### File
File operations.

```omega
let content = File.read("path/to/file")
File.write("path/to/file", "content")
File.append("path/to/file", "more content")
File.exists("path/to/file")
File.delete("path/to/file")
File.copy("src", "dst")
File.rename("old", "new")

let entries = File.read_dir("path/")
for entry in entries {
    println(entry)
}
```

### Stdin/Stdout/Stderr
Standard streams.

```omega
let input = stdin.read_line()
stdout.write("Hello")
stderr.write("Error")
```

## Math

### Basic Functions
```omega
abs(-42)         // 42
min(1, 2)        // 1
max(1, 2)        // 2
clamp(x, 0, 100)
sqrt(16)         // 4.0
cbrt(27)         // 3.0
pow(2, 10)       // 1024
exp(1)           // 2.718...
log(e)           // 1.0
log2(8)          // 3.0
log10(100)       // 2.0
```

### Trigonometric Functions
```omega
sin(PI / 2)      // 1.0
cos(0)           // 1.0
tan(PI / 4)      // 1.0
asin(1)          // PI / 2
acos(1)          // 0
atan(1)          // PI / 4
atan2(1, 1)      // PI / 4
```

### Statistics
```omega
mean([1, 2, 3, 4, 5])      // 3.0
median([1, 2, 3, 4, 5])    // 3
mode([1, 2, 2, 3, 3, 3])   // 3
variance([1, 2, 3, 4, 5])
std_dev([1, 2, 3, 4, 5])
correlation(x, y)
linear_regression(x, y)
z_score(value, mean, std_dev)
moving_average(data, window)
```

### Linear Algebra
```omega
let v = Vector::new([1.0, 2.0, 3.0])
v.dot(other)
v.cross(other)
v.normalize()
v.length()
v.project(other)

let m = Matrix::new([[1, 2], [3, 4]])
m.mul(other)
m.transpose()
m.determinant()
m.inverse()
m.trace()
```

### Number Theory
```omega
gcd(12, 8)       // 4
lcm(4, 6)        // 12
factorial(5)     // 120
fibonacci(10)    // 55
is_prime(17)     // true
primes(100)      // [2, 3, 5, 7, 11, ...]
prime_factors(60) // [2, 2, 3, 5]
```

## String

### Case Conversion
```omega
"hello".to_upper()       // "HELLO"
"HELLO".to_lower()       // "hello"
"hello world".capitalize() // "Hello world"
"helloWorld".camel_case() // "helloWorld"
"hello_world".snake_case() // "hello_world"
"helloWorld".kebab_case() // "hello-world"
```

### Manipulation
```omega
"  hello  ".trim()        // "hello"
"hello".pad_left(10, ' ') // "     hello"
"hello".pad_right(10, ' ') // "hello     "
"hello".repeat(3)         // "hellohellohello"
"hello".reverse()         // "olleh"
"hello".replace("l", "r") // "herro"
```

### Splitting and Joining
```omega
"hello,world".split(",")  // ["hello", "world"]
["hello", "world"].join(", ") // "hello, world"
"hello".chars()           // ['h', 'e', 'l', 'l', 'o']
"hello".bytes()           // [104, 101, 108, 108, 111]
```

### Pattern Matching
```omega
"hello".starts_with("he") // true
"hello".ends_with("lo")   // true
"hello".contains("ll")    // true
"hello".find("ll")        // 2
"hello".rfind("l")        // 3
```

## Network

### TCP
```omega
let listener = TcpListener::bind("127.0.0.1:8080")
let stream = listener.accept()
stream.write("Hello!")
let data = stream.read()
stream.close()
```

### UDP
```omega
let socket = UdpSocket::bind("127.0.0.1:8080")
socket.send_to("Hello!", "127.0.0.1:8081")
let (data, addr) = socket.recv_from()
```

### HTTP
```omega
let response = http.get("https://api.example.com/data")
let response = http.post("https://api.example.com/data", body)
let response = http.put("https://api.example.com/data/1", body)
let response = http.delete("https://api.example.com/data/1")
```

### URL
```omega
let url = Url::parse("https://example.com/path?query=value#fragment")
url.scheme()    // "https"
url.host()      // "example.com"
url.path()      // "/path"
url.query()     // "query=value"
url.fragment()  // "fragment"
```

## Concurrency

### Threads
```omega
let handle = thread::spawn(|| {
    println("Hello from thread!")
})
handle.join()
```

### Channels
```omega
let (tx, rx) = channel::channel()
tx.send("Hello!")
let msg = rx.recv()
tx.close()
```

### Async/Await
```omega
async fn fetch_data(url: String) -> String {
    let response = await http.get(url)
    return response.body()
}
```

### Synchronization
```omega
let mutex = Mutex::new(0)
let guard = mutex.lock()
*guard += 1
drop(guard)

let rwlock = RwLock::new(data)
let reader = rwlock.read()
let writer = rwlock.write()
```

## Serialization

### JSON
```omega
let json = Json::stringify({name: "John", age: 30})
let data = Json::parse(json)
```

### YAML
```omega
let yaml = Yaml::stringify(data)
let data = Yaml::parse(yaml)
```

### TOML
```omega
let toml = Toml::stringify(data)
let data = Toml::parse(toml)
```

### CSV
```omega
let csv = Csv::stringify(rows)
let rows = Csv::parse(csv)
```

## Testing

```omega
#[test]
fn test_addition() {
    assert_eq!(1 + 1, 2)
}

#[test]
fn test_string() {
    let s = "hello"
    assert(s.contains("ell"))
    assert_eq!(s.len(), 5)
}

#[test]
#[should_panic]
fn test_panic() {
    panic("This should panic")
}
```

## Regex

```omega
let re = Regex::new(r"\d+")
re.is_match("123")     // true
re.find("abc123def")   // "123"
re.find_all("1 2 3")   // ["1", "2", "3"]
re.replace("abc", "X") // "X"
re.split("a,b,,c")     // ["a", "b", "", "c"]
```

## Crypto

### Hashing
```omega
Hash.sha256("hello")
Hash.sha512("hello")
Hash.md5("hello")
Hash.fnv1a("hello")
Hash.crc32(data)
```

### Encoding
```omega
Base64.encode(data)
Base64.decode(encoded)
Hex.encode(data)
Hex.decode(encoded)
```

### Ciphers
```omega
Cipher.caesar_encrypt("hello", 3)
Cipher.caesar_decrypt("khoor", 3)
Cipher.rot13("hello")
Cipher.xor_encrypt(data, key)
```

## DateTime

```omega
let now = DateTime::now()
now.year()
now.month()
now.day()
now.hour()
now.minute()
now.second()
now.day_of_week()
now.format("%Y-%m-%d %H:%M:%S")

let duration = Duration::from_hours(2)
duration.as_minutes()  // 120
duration.as_seconds()  // 7200
```

## Random

```omega
let rng = Random::new()
rng.next_int()              // Random integer
rng.next_int_range(1, 100)  // 1..100
rng.next_float()            // 0.0..1.0
rng.next_bool()             // true/false
rng.next_string(10)         // Random string
rng.choose([1, 2, 3])       // Random element
rng.shuffle(array)          // In-place shuffle
rng.sample(array, 3)        // Random subset
rng.normal(0.0, 1.0)        // Normal distribution
rng.uuid_v4()               // Random UUID
```
