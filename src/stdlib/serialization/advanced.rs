use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;
use std::collections::HashMap;

// Wire type constants (protobuf-style)
const WIRE_VARINT: u8 = 0;
const WIRE_64BIT: u8 = 1;
const WIRE_LEN: u8 = 2;

// MessagePack markers
const MSG_NIL: u8 = 0xc0;
const MSG_FALSE: u8 = 0xc2;
const MSG_TRUE: u8 = 0xc3;
const MSG_INT8: u8 = 0xd0;
const MSG_INT32: u8 = 0xd2;
const MSG_INT64: u8 = 0xd3;
const MSG_UINT8: u8 = 0xcc;
const MSG_UINT32: u8 = 0xce;
const MSG_UINT64: u8 = 0xcf;
const MSG_FLOAT64: u8 = 0xcb;
const MSG_FIXSTR: u8 = 0xa0;
const MSG_STR8: u8 = 0xd9;
const MSG_FIXARRAY: u8 = 0x90;
const MSG_ARRAY16: u8 = 0xdc;
const MSG_FIXMAP: u8 = 0x80;
const MSG_MAP16: u8 = 0xde;

// Flatbuffers-style magic
const FLAT_MAGIC: [u8; 4] = *b"OMGB";

// ---------------------------------------------------------------------------
// Schema for evolution
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Int32, Int64, Float64, Bool, String,
}

#[derive(Debug, Clone)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: FieldType,
    pub field_number: u32,
}

#[derive(Debug, Clone)]
pub struct MessageSchema {
    pub name: String,
    pub version: u32,
    pub fields: Vec<FieldSchema>,
}

impl MessageSchema {
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), version: 1, fields: Vec::new() }
    }
    pub fn add_field(mut self, name: &str, ft: FieldType, num: u32) -> Self {
        self.fields.push(FieldSchema { name: name.into(), field_type: ft, field_number: num });
        self
    }
    pub fn with_version(mut self, v: u32) -> Self { self.version = v; self }
}

// ---------------------------------------------------------------------------
// Protobuf-style encoder/decoder
// ---------------------------------------------------------------------------
pub struct ProtoEncoder { buf: Vec<u8> }

impl ProtoEncoder {
    pub fn new() -> Self { Self { buf: Vec::new() } }

    pub fn encode_varint(&mut self, mut v: u64) {
        while v >= 0x80 { self.buf.push((v as u8) | 0x80); v >>= 7; }
        self.buf.push(v as u8);
    }
    pub fn encode_tag(&mut self, field: u32, wire: u8) {
        self.encode_varint(((field as u64) << 3) | wire as u64);
    }
    pub fn encode_i32_field(&mut self, f: u32, v: i32) {
        self.encode_tag(f, WIRE_VARINT); self.encode_varint(v as u32 as u64);
    }
    pub fn encode_i64_field(&mut self, f: u32, v: i64) {
        self.encode_tag(f, WIRE_VARINT); self.encode_varint(v as u64);
    }
    pub fn encode_f64_field(&mut self, f: u32, v: f64) {
        self.encode_tag(f, WIRE_64BIT); self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn encode_bool_field(&mut self, f: u32, v: bool) {
        self.encode_tag(f, WIRE_VARINT); self.buf.push(v as u8);
    }
    pub fn encode_string_field(&mut self, f: u32, v: &str) {
        self.encode_tag(f, WIRE_LEN);
        self.encode_varint(v.len() as u64);
        self.buf.extend_from_slice(v.as_bytes());
    }
    pub fn finish(self) -> Vec<u8> { self.buf }
}

pub struct ProtoDecoder<'a> { data: &'a [u8], pos: usize }

impl<'a> ProtoDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    pub fn decode_varint(&mut self) -> OmegaResult<u64> {
        let (mut result, mut shift) = (0u64, 0);
        loop {
            if self.pos >= self.data.len() {
                return Err(OmegaError::FormatError { message: "truncated varint".into() });
            }
            let byte = self.data[self.pos]; self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 { return Ok(result); }
            shift += 7;
        }
    }
    pub fn decode_tag(&mut self) -> OmegaResult<(u32, u8)> {
        let tag = self.decode_varint()?;
        Ok(((tag >> 3) as u32, (tag & 0x07) as u8))
    }
    pub fn decode_next(&mut self) -> OmegaResult<Option<(u32, Value)>> {
        if self.pos >= self.data.len() { return Ok(None); }
        let (num, wire) = self.decode_tag()?;
        match wire {
            WIRE_VARINT => Ok(Some((num, Value::Integer(self.decode_varint()? as i64)))),
            WIRE_64BIT => {
                if self.pos + 8 > self.data.len() {
                    return Err(OmegaError::FormatError { message: "truncated 64-bit".into() });
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&self.data[self.pos..self.pos+8]); self.pos += 8;
                Ok(Some((num, Value::Float(f64::from_le_bytes(b)))))
            }
            WIRE_LEN => {
                let len = self.decode_varint()? as usize;
                if self.pos + len > self.data.len() {
                    return Err(OmegaError::FormatError { message: "truncated bytes".into() });
                }
                let bytes = &self.data[self.pos..self.pos+len]; self.pos += len;
                match std::str::from_utf8(bytes) {
                    Ok(s) => Ok(Some((num, Value::String(s.into())))),
                    Err(_) => Ok(Some((num, Value::Array(bytes.iter().map(|b| Value::Integer(*b as i64)).collect())))),
                }
            }
            _ => Err(OmegaError::FormatError { message: format!("wire type {}", wire) }),
        }
    }
    pub fn decode_all(&mut self) -> OmegaResult<HashMap<u32, Value>> {
        let mut m = HashMap::new();
        while let Some((n, v)) = self.decode_next()? { m.insert(n, v); }
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// MessagePack encoder/decoder
// ---------------------------------------------------------------------------
pub struct MsgPackEncoder { buf: Vec<u8> }

impl MsgPackEncoder {
    pub fn new() -> Self { Self { buf: Vec::new() } }
    pub fn encode_nil(&mut self) { self.buf.push(MSG_NIL); }
    pub fn encode_bool(&mut self, v: bool) { self.buf.push(if v { MSG_TRUE } else { MSG_FALSE }); }
    pub fn encode_i64(&mut self, v: i64) {
        if (0..=127).contains(&v) { self.buf.push(v as u8); }
        else if (-32..0).contains(&v) { self.buf.push(v as u8); }
        else if v >= i8::MIN as i64 && v <= i8::MAX as i64 {
            self.buf.push(MSG_INT8); self.buf.push(v as i8 as u8);
        } else if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
            self.buf.push(MSG_INT32); self.buf.extend_from_slice(&(v as i32).to_be_bytes());
        } else {
            self.buf.push(MSG_INT64); self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }
    pub fn encode_f64(&mut self, v: f64) {
        self.buf.push(MSG_FLOAT64); self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub fn encode_string(&mut self, s: &str) {
        let b = s.as_bytes(); let len = b.len();
        if len <= 31 { self.buf.push(MSG_FIXSTR | len as u8); }
        else { self.buf.push(MSG_STR8); self.buf.push(len as u8); }
        self.buf.extend_from_slice(b);
    }
    pub fn encode_array_start(&mut self, len: usize) {
        if len <= 15 { self.buf.push(MSG_FIXARRAY | len as u8); }
        else { self.buf.push(MSG_ARRAY16); self.buf.extend_from_slice(&(len as u16).to_be_bytes()); }
    }
    pub fn encode_map_start(&mut self, len: usize) {
        if len <= 15 { self.buf.push(MSG_FIXMAP | len as u8); }
        else { self.buf.push(MSG_MAP16); self.buf.extend_from_slice(&(len as u16).to_be_bytes()); }
    }
    pub fn encode_value(&mut self, v: &Value) {
        match v {
            Value::None => self.encode_nil(),
            Value::Bool(b) => self.encode_bool(*b),
            Value::Integer(n) => self.encode_i64(*n),
            Value::Float(f) => self.encode_f64(*f),
            Value::String(s) => self.encode_string(s),
            Value::Array(a) => { self.encode_array_start(a.len()); a.iter().for_each(|i| self.encode_value(i)); }
            Value::Map(m) => { self.encode_map_start(m.len()); m.iter().for_each(|(k,v)| { self.encode_value(k); self.encode_value(v); }); }
            Value::Tuple(t) => { self.encode_array_start(t.len()); t.iter().for_each(|i| self.encode_value(i)); }
            _ => self.encode_string(&v.format_display()),
        }
    }
    pub fn finish(self) -> Vec<u8> { self.buf }
}

pub struct MsgPackDecoder<'a> { data: &'a [u8], pos: usize }

impl<'a> MsgPackDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn read_u8(&mut self) -> OmegaResult<u8> {
        if self.pos >= self.data.len() { return Err(OmegaError::FormatError { message: "truncated msgpack".into() }); }
        let b = self.data[self.pos]; self.pos += 1; Ok(b)
    }
    fn read_bytes(&mut self, n: usize) -> OmegaResult<&[u8]> {
        if self.pos + n > self.data.len() { return Err(OmegaError::FormatError { message: "truncated msgpack".into() }); }
        let s = &self.data[self.pos..self.pos+n]; self.pos += n; Ok(s)
    }
    pub fn decode_value(&mut self) -> OmegaResult<Value> {
        let m = self.read_u8()?;
        match m {
            MSG_NIL => Ok(Value::None),
            MSG_FALSE => Ok(Value::Bool(false)),
            MSG_TRUE => Ok(Value::Bool(true)),
            0x00..=0x7f => Ok(Value::Integer(m as i64)),
            0xe0..=0xff => Ok(Value::Integer(m as i8 as i64)),
            MSG_INT8 => Ok(Value::Integer(self.read_u8()? as i8 as i64)),
            MSG_INT32 => { let b = self.read_bytes(4)?; let mut a = [0u8;4]; a.copy_from_slice(b); Ok(Value::Integer(i32::from_be_bytes(a) as i64)) }
            MSG_INT64 => { let b = self.read_bytes(8)?; let mut a = [0u8;8]; a.copy_from_slice(b); Ok(Value::Integer(i64::from_be_bytes(a))) }
            MSG_UINT8 => Ok(Value::Integer(self.read_u8()? as i64)),
            MSG_UINT32 => { let b = self.read_bytes(4)?; let mut a = [0u8;4]; a.copy_from_slice(b); Ok(Value::Integer(u32::from_be_bytes(a) as i64)) }
            MSG_UINT64 => { let b = self.read_bytes(8)?; let mut a = [0u8;8]; a.copy_from_slice(b); Ok(Value::Integer(u64::from_be_bytes(a) as i64)) }
            MSG_FLOAT64 => { let b = self.read_bytes(8)?; let mut a = [0u8;8]; a.copy_from_slice(b); Ok(Value::Float(f64::from_be_bytes(a))) }
            0xa0..=0xbf => { let len = (m & 0x1f) as usize; Ok(Value::String(String::from_utf8_lossy(self.read_bytes(len)?).into_owned())) }
            MSG_STR8 => { let len = self.read_u8()? as usize; Ok(Value::String(String::from_utf8_lossy(self.read_bytes(len)?).into_owned())) }
            0x90..=0x9f => { let len = (m & 0x0f) as usize; (0..len).map(|_| self.decode_value()).collect::<OmegaResult<Vec<_>>>().map(Value::Array) }
            MSG_ARRAY16 => { let b = self.read_bytes(2)?; let mut a = [0u8;2]; a.copy_from_slice(b); let len = u16::from_be_bytes(a) as usize; (0..len).map(|_| self.decode_value()).collect::<OmegaResult<Vec<_>>>().map(Value::Array) }
            0x80..=0x8f => { let len = (m & 0x0f) as usize; (0..len).map(|_| Ok((self.decode_value()?, self.decode_value()?))).collect::<OmegaResult<Vec<_>>>().map(Value::Map) }
            MSG_MAP16 => { let b = self.read_bytes(2)?; let mut a = [0u8;2]; a.copy_from_slice(b); let len = u16::from_be_bytes(a) as usize; (0..len).map(|_| Ok((self.decode_value()?, self.decode_value()?))).collect::<OmegaResult<Vec<_>>>().map(Value::Map) }
            _ => Err(OmegaError::FormatError { message: format!("marker 0x{:02x}", m) }),
        }
    }
}

// ---------------------------------------------------------------------------
// Flatbuffers-style encoder/decoder
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub enum FlatValue { Int32(i32), Int64(i64), Float64(f64), Bool(bool), String(String), Bytes(Vec<u8>) }

#[derive(Debug, Clone)]
pub struct FlatTable { pub fields: Vec<Option<FlatValue>> }

impl FlatTable {
    pub fn new(size: usize) -> Self { Self { fields: vec![None; size] } }
    pub fn set(&mut self, i: usize, v: FlatValue) { if i < self.fields.len() { self.fields[i] = Some(v); } }
}

pub struct FlatEncoder;

impl FlatEncoder {
    pub fn encode_table(table: &FlatTable) -> Vec<u8> {
        let mut body = Vec::new();
        let mut vtable = vec![0u8; 4]; // size + table_size placeholders
        let fc = table.fields.len() as u16;
        vtable[0..2].copy_from_slice(&(4 + fc * 2).to_le_bytes());
        for field in &table.fields {
            if let Some(val) = field {
                vtable.extend_from_slice(&(body.len() as u16).to_le_bytes());
                match val {
                    FlatValue::Int32(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Int64(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Float64(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Bool(v) => body.push(*v as u8),
                    FlatValue::String(s) => { body.extend_from_slice(&(s.len() as u32).to_le_bytes()); body.extend_from_slice(s.as_bytes()); }
                    FlatValue::Bytes(b) => { body.extend_from_slice(&(b.len() as u32).to_le_bytes()); body.extend_from_slice(b); }
                }
            } else { vtable.extend_from_slice(&0u16.to_le_bytes()); }
        }
        vtable[2..4].copy_from_slice(&(body.len() as u16).to_le_bytes());
        let mut out = Vec::with_capacity(4 + vtable.len() + body.len());
        out.extend_from_slice(&FLAT_MAGIC);
        out.extend_from_slice(&(vtable.len() as u16).to_le_bytes());
        out.extend_from_slice(&vtable);
        out.extend_from_slice(&body);
        out
    }
}

pub fn flat_decode_table(data: &[u8]) -> OmegaResult<FlatTable> {
    if data.len() < 8 || data[0..4] != FLAT_MAGIC {
        return Err(OmegaError::FormatError { message: "invalid flatbuffers".into() });
    }
    let vs = u16::from_le_bytes([data[4], data[5]]) as usize;
    let fc = (vs - 4) / 2;
    let vstart = 8; let bstart = vstart + vs;
    let mut table = FlatTable::new(fc);
    for i in 0..fc {
        let op = vstart + 4 + i * 2;
        if op + 2 > data.len() { break; }
        let off = u16::from_le_bytes([data[op], data[op+1]]) as usize;
        if off == 0 { continue; }
        let abs = bstart + off;
        if abs + 4 <= data.len() {
            let mut buf = [0u8; 4]; buf.copy_from_slice(&data[abs..abs+4]);
            table.fields[i] = Some(FlatValue::Int32(i32::from_le_bytes(buf)));
        }
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// Schema evolution
// ---------------------------------------------------------------------------
pub fn evolve_schema(
    data: &HashMap<u32, Value>, old: &MessageSchema, new: &MessageSchema,
) -> OmegaResult<HashMap<u32, Value>> {
    let old_f: HashMap<u32, &FieldSchema> = old.fields.iter().map(|f| (f.field_number, f)).collect();
    let new_f: HashMap<u32, &FieldSchema> = new.fields.iter().map(|f| (f.field_number, f)).collect();
    let mut result = HashMap::new();
    for (&num, val) in data {
        if let Some(nf) = new_f.get(&num) {
            if let Some(of) = old_f.get(&num) {
                result.insert(num, convert_value(val, &of.field_type, &nf.field_type)?);
            }
        }
    }
    for (&num, field) in &new_f {
        if !result.contains_key(&num) {
            result.insert(num, default_value(&field.field_type));
        }
    }
    Ok(result)
}

fn convert_value(v: &Value, from: &FieldType, to: &FieldType) -> OmegaResult<Value> {
    match (from, to) {
        (FieldType::Int32, FieldType::Int64) | (FieldType::Int32, FieldType::Float64) | (FieldType::Int64, FieldType::Float64) => {
            match v { Value::Integer(n) => if *to == FieldType::Float64 { Ok(Value::Float(*n as f64)) } else { Ok(Value::Integer(*n)) }, _ => Ok(v.clone()) }
        }
        _ if from == to => Ok(v.clone()),
        _ => Err(OmegaError::FormatError { message: format!("{:?} -> {:?}", from, to) }),
    }
}

fn default_value(ft: &FieldType) -> Value {
    match ft {
        FieldType::Int32 | FieldType::Int64 => Value::Integer(0),
        FieldType::Float64 => Value::Float(0.0),
        FieldType::Bool => Value::Bool(false),
        FieldType::String => Value::String(String::new()),
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------
pub fn proto_encode_value(v: &Value) -> OmegaResult<Vec<u8>> {
    let mut e = ProtoEncoder::new();
    match v {
        Value::Integer(n) => e.encode_i64_field(1, *n),
        Value::Float(f) => e.encode_f64_field(2, *f),
        Value::String(s) => e.encode_string_field(3, s),
        Value::Bool(b) => e.encode_bool_field(4, *b),
        _ => e.encode_string_field(5, &v.format_display()),
    }
    Ok(e.finish())
}

pub fn msgpack_encode(v: &Value) -> Vec<u8> {
    let mut e = MsgPackEncoder::new(); e.encode_value(v); e.finish()
}
pub fn msgpack_decode(d: &[u8]) -> OmegaResult<Value> {
    MsgPackDecoder::new(d).decode_value()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_varint_roundtrip() {
        let mut e = ProtoEncoder::new(); e.encode_varint(150);
        let mut d = ProtoDecoder::new(&e.finish());
        assert_eq!(d.decode_varint().unwrap(), 150);
    }

    #[test]
    fn test_proto_string_field() {
        let mut e = ProtoEncoder::new(); e.encode_string_field(1, "hello");
        let mut d = ProtoDecoder::new(&e.finish());
        let (n, v) = d.decode_next().unwrap().unwrap();
        assert_eq!(n, 1); assert_eq!(v, Value::String("hello".into()));
    }

    #[test]
    fn test_proto_multiple_fields() {
        let mut e = ProtoEncoder::new();
        e.encode_i32_field(1, 42); e.encode_f64_field(2, 3.14); e.encode_string_field(3, "x");
        let f = ProtoDecoder::new(&e.finish()).decode_all().unwrap();
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn test_msgpack_nil_and_bool() {
        let mut e = MsgPackEncoder::new(); e.encode_nil(); e.encode_bool(true); e.encode_bool(false);
        let mut d = MsgPackDecoder::new(&e.finish());
        assert_eq!(d.decode_value().unwrap(), Value::None);
        assert_eq!(d.decode_value().unwrap(), Value::Bool(true));
        assert_eq!(d.decode_value().unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_msgpack_integer_range() {
        let encoded = { let mut e = MsgPackEncoder::new(); e.encode_i64(0); e.encode_i64(127); e.encode_i64(-32); e.encode_i64(300); e.encode_i64(-300); e.finish() };
        let mut d = MsgPackDecoder::new(&encoded);
        for expected in [0, 127, -32, 300, -300] {
            assert_eq!(d.decode_value().unwrap(), Value::Integer(expected));
        }
    }

    #[test]
    fn test_msgpack_string_roundtrip() {
        for s in ["hi", &"a".repeat(100)] {
            let v = Value::String(s.to_string());
            assert_eq!(msgpack_decode(&msgpack_encode(&v)).unwrap(), v);
        }
    }

    #[test]
    fn test_msgpack_array_roundtrip() {
        let arr = Value::Array(vec![Value::Integer(1), Value::String("two".into()), Value::Float(3.0)]);
        assert_eq!(msgpack_decode(&msgpack_encode(&arr)).unwrap(), arr);
    }

    #[test]
    fn test_msgpack_map_roundtrip() {
        let map = Value::Map(vec![(Value::String("a".into()), Value::Integer(1)), (Value::String("b".into()), Value::Bool(true))]);
        assert_eq!(msgpack_decode(&msgpack_encode(&map)).unwrap(), map);
    }

    #[test]
    fn test_msgpack_nested() {
        let nested = Value::Array(vec![
            Value::Map(vec![(Value::String("id".into()), Value::Integer(1)),
                            (Value::String("tags".into()), Value::Array(vec![Value::String("a".into())]))]),
        ]);
        assert_eq!(msgpack_decode(&msgpack_encode(&nested)).unwrap(), nested);
    }

    #[test]
    fn test_flatbuffers_roundtrip() {
        let mut t = FlatTable::new(3);
        t.set(0, FlatValue::Int32(42));
        t.set(1, FlatValue::String("hello".into()));
        t.set(2, FlatValue::Bool(true));
        let data = FlatEncoder::encode_table(&t);
        assert_eq!(&data[0..4], &FLAT_MAGIC);
        let dec = flat_decode_table(&data).unwrap();
        assert!(matches!(dec.fields[0], Some(FlatValue::Int32(42))));
    }

    #[test]
    fn test_schema_evolution_add_field() {
        let old = MessageSchema::new("User").add_field("name", FieldType::String, 1);
        let new = MessageSchema::new("User").add_field("name", FieldType::String, 1).add_field("email", FieldType::String, 2);
        let mut data = HashMap::new(); data.insert(1, Value::String("alice".into()));
        let evolved = evolve_schema(&data, &old, &new).unwrap();
        assert_eq!(evolved.get(&1), Some(&Value::String("alice".into())));
        assert_eq!(evolved.get(&2), Some(&Value::String(String::new())));
    }

    #[test]
    fn test_schema_evolution_widen_type() {
        let old = MessageSchema::new("M").add_field("v", FieldType::Int32, 1);
        let new = MessageSchema::new("M").add_field("v", FieldType::Float64, 1);
        let mut data = HashMap::new(); data.insert(1, Value::Integer(42));
        let evolved = evolve_schema(&data, &old, &new).unwrap();
        assert_eq!(evolved.get(&1), Some(&Value::Float(42.0)));
    }

    #[test]
    fn test_proto_encode_convenience() {
        let v = Value::Map(vec![(Value::String("x".into()), Value::Integer(10))]);
        let encoded = proto_encode_value(&v).unwrap();
        assert!(!ProtoDecoder::new(&encoded).decode_all().unwrap().is_empty());
    }
}
