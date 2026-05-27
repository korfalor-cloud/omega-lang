use omega_lang::stdlib::compression::huffman::HuffmanCoder;
use omega_lang::stdlib::compression::run_length::RunLengthCoder;

#[test]
fn test_huffman_encode_decode() {
    let data = b"hello world";
    let mut coder = HuffmanCoder::new();
    coder.build(data);

    let encoded = coder.encode(data);
    let decoded = coder.decode(&encoded);

    assert_eq!(decoded, data);
}

#[test]
fn test_huffman_empty() {
    let data = b"";
    let mut coder = HuffmanCoder::new();
    coder.build(data);

    let encoded = coder.encode(data);
    assert!(encoded.is_empty());
}

#[test]
fn test_huffman_single_char() {
    let data = b"aaaa";
    let mut coder = HuffmanCoder::new();
    coder.build(data);

    let encoded = coder.encode(data);
    let decoded = coder.decode(&encoded);

    assert_eq!(decoded, data);
}

#[test]
fn test_huffman_compression_ratio() {
    let data = b"aaaaabbbbbcccccdddddeeeee";
    let mut coder = HuffmanCoder::new();
    coder.build(data);

    let encoded = coder.encode(data);
    let ratio = coder.compression_ratio(data, &encoded);

    // Should achieve some compression
    assert!(ratio < 1.0);
}

#[test]
fn test_huffman_codes() {
    let data = b"hello";
    let mut coder = HuffmanCoder::new();
    coder.build(data);

    let codes = coder.codes();
    assert!(!codes.is_empty());

    // Most frequent character should have shortest code
    // 'l' appears twice, others once
    let l_code = codes.get(&b'l').unwrap();
    assert!(l_code.len() <= 2);
}

#[test]
fn test_run_length_encode() {
    let data = b"aaabbbcccc";
    let encoded = RunLengthCoder::encode(data);

    // Should be shorter than original
    assert!(encoded.len() < data.len());
}

#[test]
fn test_run_length_decode() {
    let data = b"aaabbbcccc";
    let encoded = RunLengthCoder::encode(data);
    let decoded = RunLengthCoder::decode(&encoded);

    assert_eq!(decoded, data);
}

#[test]
fn test_run_length_empty() {
    let data = b"";
    let encoded = RunLengthCoder::encode(data);
    assert!(encoded.is_empty());
}

#[test]
fn test_run_length_no_repeats() {
    let data = b"abcdef";
    let encoded = RunLengthCoder::encode(data);

    // No repeats means 2 bytes per char
    assert_eq!(encoded.len(), data.len() * 2);
}

#[test]
fn test_run_length_all_same() {
    let data = b"aaaa";
    let encoded = RunLengthCoder::encode(data);

    // All same means 2 bytes total
    assert_eq!(encoded.len(), 2);
}

#[test]
fn test_run_length_encode_decode_roundtrip() {
    let data = b"aabbbccccddddeeeeeeffff";
    let encoded = RunLengthCoder::encode(data);
    let decoded = RunLengthCoder::decode(&encoded);
    assert_eq!(decoded, data);
}

#[test]
fn test_run_length_string() {
    let s = "aaabbbcccc";
    let encoded = RunLengthCoder::encode_string(s);
    let decoded = RunLengthCoder::decode_string(&encoded);
    assert_eq!(decoded, s);
}

#[test]
fn test_run_length_compression_ratio() {
    let data = b"aaabbbcccc";
    let encoded = RunLengthCoder::encode(data);
    let ratio = RunLengthCoder::compression_ratio(data, &encoded);

    // Should achieve compression with repeated data
    assert!(ratio < 1.0);
}
