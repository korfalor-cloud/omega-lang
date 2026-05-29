use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Wire type constants (protobuf-style)
// ---------------------------------------------------------------------------
const WIRE_VARINT: u8 = 0;
const WIRE_64BIT: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_32BIT: u8 = 5;

// MessagePack type markers
const MSG_NIL: u8 = 0xc0;
const MSG_FALSE: u8 = 0xc2;
const MSG_TRUE: u8 = 0xc3;
const MSG_FIXINT: u8 = 0x00;
const MSG_UINT8: u8 = 0xcc;
const MSG_UINT32: u8 = 0xce;
const MSG_UINT64: u8 = 0xcf;
const MSG_INT8: u8 = 0xd0;
const MSG_INT32: u8 = 0xd2;
const MSG_INT64: u8 = 0xd3;
const MSG_FLOAT64: u8 = 0xcb;
const MSG_FIXSTR: u8 = 0xa0;
const MSG_STR8: u8 = 0xd9;
const MSG_STR32: u8 = 0xdb;
const MSG_FIXARRAY: u8 = 0x90;
const MSG_ARRAY16: u8 = 0xdc;
const MSG_FIXMAP: u8 = 0x80;
const MSG_MAP16: u8 = 0xde;

// Flatbuffers-style header
const FLAT_MAGIC: [u8; 4] = [0x4f, 0x4d, 0x47, 0x42]; // "OMGB"

// ---------------------------------------------------------------------------
// Schema definition for evolution support
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: FieldType,
    pub field_number: u32,
    pub optional: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Int32,
    Int64,
    Float64,
    Bool,
    String,
    Bytes,
    Message(String),
    Array(Box<FieldType>),
    Map(Box<FieldType>, Box<FieldType>),
}

#[derive(Debug, Clone)]
pub struct MessageSchema {
    pub name: String,
    pub version: u32,
    pub fields: Vec<FieldSchema>,
}

impl MessageSchema {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: 1,
            fields: Vec::new(),
        }
    }

    pub fn add_field(mut self, name: &str, ft: FieldType, number: u32) -> Self {
        self.fields.push(FieldSchema {
            name: name.to_string(),
            field_type: ft,
            field_number: number,
            optional: true,
        });
        self
    }

    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }
}

// ---------------------------------------------------------------------------
// Protobuf-style encoding
// ---------------------------------------------------------------------------
pub struct ProtoEncoder {
    buffer: Vec<u8>,
}

impl ProtoEncoder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn encode_varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.buffer.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.buffer.push(value as u8);
    }

    pub fn encode_tag(&mut self, field_number: u32, wire_type: u8) {
        self.encode_varint(((field_number as u64) << 3) | (wire_type as u64));
    }

    pub fn encode_i32_field(&mut self, field_number: u32, value: i32) {
        self.encode_tag(field_number, WIRE_VARINT);
        self.encode_varint(value as u32 as u64);
    }

    pub fn encode_i64_field(&mut self, field_number: u32, value: i64) {
        self.encode_tag(field_number, WIRE_VARINT);
        self.encode_varint(value as u64);
    }

    pub fn encode_f64_field(&mut self, field_number: u32, value: f64) {
        self.encode_tag(field_number, WIRE_64BIT);
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    pub fn encode_bool_field(&mut self, field_number: u32, value: bool) {
        self.encode_tag(field_number, WIRE_VARINT);
        self.buffer.push(value as u8);
    }

    pub fn encode_string_field(&mut self, field_number: u32, value: &str) {
        self.encode_tag(field_number, WIRE_LEN);
        let bytes = value.as_bytes();
        self.encode_varint(bytes.len() as u64);
        self.buffer.extend_from_slice(bytes);
    }

    pub fn encode_bytes_field(&mut self, field_number: u32, data: &[u8]) {
        self.encode_tag(field_number, WIRE_LEN);
        self.encode_varint(data.len() as u64);
        self.buffer.extend_from_slice(data);
    }

    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }
}

pub struct ProtoDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn decode_varint(&mut self) -> OmegaResult<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(OmegaError::FormatError {
                    message: "unexpected end of varint".into(),
                });
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    pub fn decode_tag(&mut self) -> OmegaResult<(u32, u8)> {
        let tag = self.decode_varint()?;
        Ok(((tag >> 3) as u32, (tag & 0x07) as u8))
    }

    pub fn decode_next(&mut self) -> OmegaResult<Option<(u32, Value)>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let (field_number, wire_type) = self.decode_tag()?;
        match wire_type {
            WIRE_VARINT => {
                let v = self.decode_varint()?;
                Ok(Some((field_number, Value::Integer(v as i64))))
            }
            WIRE_64BIT => {
                if self.pos + 8 > self.data.len() {
                    return Err(OmegaError::FormatError {
                        message: "unexpected end of 64-bit value".into(),
                    });
                }
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&self.data[self.pos..self.pos + 8]);
                self.pos += 8;
                Ok(Some((field_number, Value::Float(f64::from_le_bytes(buf)))))
            }
            WIRE_LEN => {
                let len = self.decode_varint()? as usize;
                if self.pos + len > self.data.len() {
                    return Err(OmegaError::FormatError {
                        message: "unexpected end of length-delimited value".into(),
                    });
                }
                let bytes = &self.data[self.pos..self.pos + len];
                self.pos += len;
                match std::str::from_utf8(bytes) {
                    Ok(s) => Ok(Some((field_number, Value::String(s.to_string())))),
                    Err(_) => {
                        Ok(Some((field_number, Value::Array(
                            bytes.iter().map(|b| Value::Integer(*b as i64)).collect(),
                        ))))
                    }
                }
            }
            _ => Err(OmegaError::FormatError {
                message: format!("unsupported wire type: {}", wire_type),
            }),
        }
    }

    pub fn decode_all(&mut self) -> OmegaResult<HashMap<u32, Value>> {
        let mut fields = HashMap::new();
        while let Some((num, val)) = self.decode_next()? {
            fields.insert(num, val);
        }
        Ok(fields)
    }
}

// ---------------------------------------------------------------------------
// MessagePack encoding
// ---------------------------------------------------------------------------
pub struct MsgPackEncoder {
    buffer: Vec<u8>,
}

impl MsgPackEncoder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn encode_nil(&mut self) {
        self.buffer.push(MSG_NIL);
    }

    pub fn encode_bool(&mut self, value: bool) {
        self.buffer.push(if value { MSG_TRUE } else { MSG_FALSE });
    }

    pub fn encode_i64(&mut self, value: i64) {
        if value >= 0 && value <= 127 {
            self.buffer.push(value as u8);
        } else if value >= -32 && value < 0 {
            self.buffer.push(value as u8);
        } else if value >= i8::MIN as i64 && value <= i8::MAX as i64 {
            self.buffer.push(MSG_INT8);
            self.buffer.push(value as i8 as u8);
        } else if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
            self.buffer.push(MSG_INT32);
            self.buffer.extend_from_slice(&(value as i32).to_be_bytes());
        } else {
            self.buffer.push(MSG_INT64);
            self.buffer.extend_from_slice(&value.to_be_bytes());
        }
    }

    pub fn encode_f64(&mut self, value: f64) {
        self.buffer.push(MSG_FLOAT64);
        self.buffer.extend_from_slice(&value.to_be_bytes());
    }

    pub fn encode_string(&mut self, value: &str) {
        let bytes = value.as_bytes();
        let len = bytes.len();
        if len <= 31 {
            self.buffer.push(MSG_FIXSTR | len as u8);
        } else if len <= 255 {
            self.buffer.push(MSG_STR8);
            self.buffer.push(len as u8);
        } else {
            self.buffer.push(MSG_STR32);
            self.buffer.extend_from_slice(&(len as u32).to_be_bytes());
        }
        self.buffer.extend_from_slice(bytes);
    }

    pub fn encode_array_start(&mut self, len: usize) {
        if len <= 15 {
            self.buffer.push(MSG_FIXARRAY | len as u8);
        } else {
            self.buffer.push(MSG_ARRAY16);
            self.buffer.extend_from_slice(&(len as u16).to_be_bytes());
        }
    }

    pub fn encode_map_start(&mut self, len: usize) {
        if len <= 15 {
            self.buffer.push(MSG_FIXMAP | len as u8);
        } else {
            self.buffer.push(MSG_MAP16);
            self.buffer.extend_from_slice(&(len as u16).to_be_bytes());
        }
    }

    pub fn encode_value(&mut self, value: &Value) {
        match value {
            Value::None => self.encode_nil(),
            Value::Bool(b) => self.encode_bool(*b),
            Value::Integer(n) => self.encode_i64(*n),
            Value::Float(f) => self.encode_f64(*f),
            Value::String(s) => self.encode_string(s),
            Value::Array(arr) => {
                self.encode_array_start(arr.len());
                for item in arr {
                    self.encode_value(item);
                }
            }
            Value::Map(map) => {
                self.encode_map_start(map.len());
                for (k, v) in map {
                    self.encode_value(k);
                    self.encode_value(v);
                }
            }
            Value::Tuple(t) => {
                self.encode_array_start(t.len());
                for item in t {
                    self.encode_value(item);
                }
            }
            _ => {
                self.encode_string(&value.format_display());
            }
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }
}

pub struct MsgPackDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MsgPackDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> OmegaResult<u8> {
        if self.pos >= self.data.len() {
            return Err(OmegaError::FormatError {
                message: "unexpected end of msgpack data".into(),
            });
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> OmegaResult<&[u8]> {
        if self.pos + n > self.data.len() {
            return Err(OmegaError::FormatError {
                message: "unexpected end of msgpack data".into(),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn decode_value(&mut self) -> OmegaResult<Value> {
        let marker = self.read_u8()?;
        match marker {
            MSG_NIL => Ok(Value::None),
            MSG_FALSE => Ok(Value::Bool(false)),
            MSG_TRUE => Ok(Value::Bool(true)),
            0x00..=0x7f => Ok(Value::Integer(marker as i64)),
            0xe0..=0xff => Ok(Value::Integer(marker as i8 as i64)),
            MSG_INT8 => {
                let b = self.read_u8()?;
                Ok(Value::Integer(b as i8 as i64))
            }
            MSG_INT32 => {
                let bytes = self.read_bytes(4)?;
                let mut buf = [0u8; 4];
                buf.copy_from_slice(bytes);
                Ok(Value::Integer(i32::from_be_bytes(buf) as i64))
            }
            MSG_INT64 => {
                let bytes = self.read_bytes(8)?;
                let mut buf = [0u8; 8];
                buf.copy_from_slice(bytes);
                Ok(Value::Integer(i64::from_be_bytes(buf)))
            }
            MSG_UINT8 => {
                let b = self.read_u8()?;
                Ok(Value::Integer(b as i64))
            }
            MSG_UINT32 => {
                let bytes = self.read_bytes(4)?;
                let mut buf = [0u8; 4];
                buf.copy_from_slice(bytes);
                Ok(Value::Integer(u32::from_be_bytes(buf) as i64))
            }
            MSG_UINT64 => {
                let bytes = self.read_bytes(8)?;
                let mut buf = [0u8; 8];
                buf.copy_from_slice(bytes);
                Ok(Value::Integer(u64::from_be_bytes(buf) as i64))
            }
            MSG_FLOAT64 => {
                let bytes = self.read_bytes(8)?;
                let mut buf = [0u8; 8];
                buf.copy_from_slice(bytes);
                Ok(Value::Float(f64::from_be_bytes(buf)))
            }
            0xa0..=0xbf => {
                let len = (marker & 0x1f) as usize;
                let bytes = self.read_bytes(len)?;
                Ok(Value::String(String::from_utf8_lossy(bytes).into_owned()))
            }
            MSG_STR8 => {
                let len = self.read_u8()? as usize;
                let bytes = self.read_bytes(len)?;
                Ok(Value::String(String::from_utf8_lossy(bytes).into_owned()))
            }
            MSG_STR32 => {
                let b = self.read_bytes(4)?;
                let mut buf = [0u8; 4];
                buf.copy_from_slice(b);
                let len = u32::from_be_bytes(buf) as usize;
                let bytes = self.read_bytes(len)?;
                Ok(Value::String(String::from_utf8_lossy(bytes).into_owned()))
            }
            0x90..=0x9f => {
                let len = (marker & 0x0f) as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.decode_value()?);
                }
                Ok(Value::Array(arr))
            }
            MSG_ARRAY16 => {
                let b = self.read_bytes(2)?;
                let mut buf = [0u8; 2];
                buf.copy_from_slice(b);
                let len = u16::from_be_bytes(buf) as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.decode_value()?);
                }
                Ok(Value::Array(arr))
            }
            0x80..=0x8f => {
                let len = (marker & 0x0f) as usize;
                let mut map = Vec::with_capacity(len);
                for _ in 0..len {
                    let k = self.decode_value()?;
                    let v = self.decode_value()?;
                    map.push((k, v));
                }
                Ok(Value::Map(map))
            }
            MSG_MAP16 => {
                let b = self.read_bytes(2)?;
                let mut buf = [0u8; 2];
                buf.copy_from_slice(b);
                let len = u16::from_be_bytes(buf) as usize;
                let mut map = Vec::with_capacity(len);
                for _ in 0..len {
                    let k = self.decode_value()?;
                    let v = self.decode_value()?;
                    map.push((k, v));
                }
                Ok(Value::Map(map))
            }
            _ => Err(OmegaError::FormatError {
                message: format!("unsupported msgpack marker: 0x{:02x}", marker),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Flatbuffers-style encoding (table-based, with vtable)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FlatValue {
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Table(FlatTable),
}

#[derive(Debug, Clone)]
pub struct FlatTable {
    pub fields: Vec<Option<FlatValue>>,
}

impl FlatTable {
    pub fn new(size: usize) -> Self {
        Self {
            fields: vec![None; size],
        }
    }

    pub fn set(&mut self, index: usize, value: FlatValue) {
        if index < self.fields.len() {
            self.fields[index] = Some(value);
        }
    }
}

pub struct FlatEncoder {
    buffer: Vec<u8>,
}

impl FlatEncoder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn encode_table(&mut self, table: &FlatTable) -> Vec<u8> {
        let mut body = Vec::new();

        // vtable: field offsets relative to vtable start
        let mut vtable = Vec::new();
        vtable.extend_from_slice(&(0u16).to_le_bytes()); // vtable size placeholder
        vtable.extend_from_slice(&(0u16).to_le_bytes()); // table size placeholder

        let field_count = table.fields.len() as u16;
        vtable[0..2].copy_from_slice(&(4 + field_count * 2).to_le_bytes());

        for field in &table.fields {
            if let Some(val) = field {
                let offset = body.len() as u16;
                vtable.extend_from_slice(&offset.to_le_bytes());
                match val {
                    FlatValue::Int32(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Int64(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Float64(v) => body.extend_from_slice(&v.to_le_bytes()),
                    FlatValue::Bool(v) => body.push(*v as u8),
                    FlatValue::String(s) => {
                        let bytes = s.as_bytes();
                        body.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        body.extend_from_slice(bytes);
                    }
                    FlatValue::Bytes(b) => {
                        body.extend_from_slice(&(b.len() as u32).to_le_bytes());
                        body.extend_from_slice(b);
                    }
                    FlatValue::Table(t) => {
                        let nested = self.encode_table(t);
                        body.extend_from_slice(&(nested.len() as u32).to_le_bytes());
                        body.extend_from_slice(&nested);
                    }
                }
            } else {
                vtable.extend_from_slice(&0u16.to_le_bytes());
            }
        }

        let total_size = body.len() as u16;
        vtable[2..4].copy_from_slice(&total_size.to_le_bytes());

        let mut result = Vec::new();
        result.extend_from_slice(&FLAT_MAGIC);
        result.extend_from_slice(&(vtable.len() as u16).to_le_bytes());
        result.extend_from_slice(&vtable);
        result.extend_from_slice(&body);
        result
    }
}

pub fn flat_decode_table(data: &[u8]) -> OmegaResult<FlatTable> {
    if data.len() < 4 {
        return Err(OmegaError::FormatError {
            message: "flatbuffers data too short".into(),
        });
    }
    if data[0..4] != FLAT_MAGIC {
        return Err(OmegaError::FormatError {
            message: "invalid flatbuffers magic bytes".into(),
        });
    }

    let vtable_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let table_size = u16::from_le_bytes([data[6], data[7]]) as usize;
    let field_count = (vtable_size - 4) / 2;

    let mut table = FlatTable::new(field_count);
    let vtable_start = 8;
    let body_start = vtable_start + vtable_size;

    for i in 0..field_count {
        let offset_pos = vtable_start + 4 + i * 2;
        if offset_pos + 2 > data.len() {
            break;
        }
        let field_offset = u16::from_le_bytes([data[offset_pos], data[offset_pos + 1]]) as usize;
        if field_offset == 0 {
            continue;
        }
        let abs_offset = body_start + field_offset;
        if abs_offset + 4 <= data.len() {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&data[abs_offset..abs_offset + 4]);
            table.fields[i] = Some(FlatValue::Int32(i32::from_le_bytes(buf)));
        }
    }

    table.fields.truncate(field_count);
    Ok(table)
}

// ---------------------------------------------------------------------------
// Schema evolution: migrate data between schema versions
// ---------------------------------------------------------------------------
pub fn evolve_schema(
    data: &HashMap<u32, Value>,
    old_schema: &MessageSchema,
    new_schema: &MessageSchema,
) -> OmegaResult<HashMap<u32, Value>> {
    let old_fields: HashMap<u32, &FieldSchema> =
        old_schema.fields.iter().map(|f| (f.field_number, f)).collect();
    let new_fields: HashMap<u32, &FieldSchema> =
        new_schema.fields.iter().map(|f| (f.field_number, f)).collect();

    let mut result = HashMap::new();

    for (field_num, value) in data {
        if let Some(new_field) = new_fields.get(field_num) {
            if let Some(old_field) = old_fields.get(field_num) {
                let converted = convert_value(value, &old_field.field_type, &new_field.field_type)?;
                result.insert(*field_num, converted);
            }
        }
        // Fields not in new schema are silently dropped (forward compat)
    }

    // Fill defaults for fields added in the new schema
    for (num, field) in &new_fields {
        if !result.contains_key(num) {
            let default = default_value(&field.field_type);
            result.insert(*num, default);
        }
    }

    Ok(result)
}

fn convert_value(value: &Value, from: &FieldType, to: &FieldType) -> OmegaResult<Value> {
    match (from, to) {
        (FieldType::Int32, FieldType::Int64) => {
            if let Value::Integer(n) = value {
                Ok(Value::Integer(*n))
            } else {
                Ok(value.clone())
            }
        }
        (FieldType::Int32, FieldType::Float64) => {
            if let Value::Integer(n) = value {
                Ok(Value::Float(*n as f64))
            } else {
                Ok(value.clone())
            }
        }
        (FieldType::Int64, FieldType::Float64) => {
            if let Value::Integer(n) = value {
                Ok(Value::Float(*n as f64))
            } else {
                Ok(value.clone())
            }
        }
        _ if from == to => Ok(value.clone()),
        _ => Err(OmegaError::FormatError {
            message: format!(
                "incompatible field type change from {:?} to {:?}",
                from, to
            ),
        }),
    }
}

fn default_value(ft: &FieldType) -> Value {
    match ft {
        FieldType::Int32 | FieldType::Int64 => Value::Integer(0),
        FieldType::Float64 => Value::Float(0.0),
        FieldType::Bool => Value::Bool(false),
        FieldType::String => Value::String(String::new()),
        FieldType::Bytes => Value::Array(Vec::new()),
        _ => Value::None,
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------
pub fn proto_encode_value(value: &Value) -> OmegaResult<Vec<u8>> {
    let mut encoder = ProtoEncoder::new();
    match value {
        Value::Integer(n) => encoder.encode_i64_field(1, *n),
        Value::Float(f) => encoder.encode_f64_field(2, *f),
        Value::String(s) => encoder.encode_string_field(3, s),
        Value::Bool(b) => encoder.encode_bool_field(4, *b),
        Value::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                match item {
                    Value::Integer(n) => encoder.encode_i64_field(5, *n),
                    Value::Float(f) => encoder.encode_f64_field(5, *f),
                    Value::String(s) => encoder.encode_string_field(5, s),
                    _ => encoder.encode_string_field(5, &item.format_display()),
                }
            }
        }
        Value::Map(map) => {
            for (k, v) in map {
                if let Value::String(key) = k {
                    match v {
                        Value::Integer(n) => encoder.encode_i64_field(6, *n),
                        Value::Float(f) => encoder.encode_f64_field(6, *f),
                        Value::String(s) => encoder.encode_string_field(6, s),
                        Value::Bool(b) => encoder.encode_bool_field(6, *b),
                        _ => encoder.encode_string_field(6, &v.format_display()),
                    }
                }
            }
        }
        _ => encoder.encode_string_field(7, &value.format_display()),
    }
    Ok(encoder.finish())
}

pub fn msgpack_encode(value: &Value) -> Vec<u8> {
    let mut encoder = MsgPackEncoder::new();
    encoder.encode_value(value);
    encoder.finish()
}

pub fn msgpack_decode(data: &[u8]) -> OmegaResult<Value> {
    let mut decoder = MsgPackDecoder::new(data);
    decoder.decode_value()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_varint_roundtrip() {
        let mut enc = ProtoEncoder::new();
        enc.encode_varint(150);
        let data = enc.finish();
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.decode_varint().unwrap(), 150);
    }

    #[test]
    fn test_proto_string_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_string_field(1, "hello world");
        let data = enc.finish();
        let mut dec = ProtoDecoder::new(&data);
        let (num, val) = dec.decode_next().unwrap().unwrap();
        assert_eq!(num, 1);
        assert_eq!(val, Value::String("hello world".into()));
    }

    #[test]
    fn test_proto_multiple_fields() {
        let mut enc = ProtoEncoder::new();
        enc.encode_i32_field(1, 42);
        enc.encode_f64_field(2, 3.14);
        enc.encode_string_field(3, "test");
        let data = enc.finish();
        let mut dec = ProtoDecoder::new(&data);
        let fields = dec.decode_all().unwrap();
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn test_msgpack_nil_and_bool() {
        let encoded = {
            let mut enc = MsgPackEncoder::new();
            enc.encode_nil();
            enc.encode_bool(true);
            enc.encode_bool(false);
            enc.finish()
        };
        let mut dec = MsgPackDecoder::new(&encoded);
        assert_eq!(dec.decode_value().unwrap(), Value::None);
        assert_eq!(dec.decode_value().unwrap(), Value::Bool(true));
        assert_eq!(dec.decode_value().unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_msgpack_integer_range() {
        let encoded = {
            let mut enc = MsgPackEncoder::new();
            enc.encode_i64(0);
            enc.encode_i64(127);
            enc.encode_i64(-32);
            enc.encode_i64(300);
            enc.encode_i64(-300);
            enc.finish()
        };
        let mut dec = MsgPackDecoder::new(&encoded);
        assert_eq!(dec.decode_value().unwrap(), Value::Integer(0));
        assert_eq!(dec.decode_value().unwrap(), Value::Integer(127));
        assert_eq!(dec.decode_value().unwrap(), Value::Integer(-32));
        assert_eq!(dec.decode_value().unwrap(), Value::Integer(300));
        assert_eq!(dec.decode_value().unwrap(), Value::Integer(-300));
    }

    #[test]
    fn test_msgpack_string_roundtrip() {
        let short = "hi";
        let medium = "a".repeat(100);
        let values = vec![
            Value::String(short.to_string()),
            Value::String(medium.clone()),
        ];
        for val in &values {
            let encoded = msgpack_encode(val);
            let decoded = msgpack_decode(&encoded).unwrap();
            assert_eq!(*val, decoded);
        }
    }

    #[test]
    fn test_msgpack_array_roundtrip() {
        let arr = Value::Array(vec![
            Value::Integer(1),
            Value::String("two".into()),
            Value::Float(3.0),
        ]);
        let encoded = msgpack_encode(&arr);
        let decoded = msgpack_decode(&encoded).unwrap();
        assert_eq!(arr, decoded);
    }

    #[test]
    fn test_msgpack_map_roundtrip() {
        let map = Value::Map(vec![
            (Value::String("a".into()), Value::Integer(1)),
            (Value::String("b".into()), Value::Bool(true)),
        ]);
        let encoded = msgpack_encode(&map);
        let decoded = msgpack_decode(&encoded).unwrap();
        assert_eq!(map, decoded);
    }

    #[test]
    fn test_flatbuffers_encode_decode() {
        let mut table = FlatTable::new(3);
        table.set(0, FlatValue::Int32(42));
        table.set(1, FlatValue::String("hello".into()));
        table.set(2, FlatValue::Bool(true));

        let mut encoder = FlatEncoder::new();
        let data = encoder.encode_table(&table);
        assert!(&data[0..4] == &FLAT_MAGIC);

        let decoded = flat_decode_table(&data).unwrap();
        assert!(matches!(decoded.fields[0], Some(FlatValue::Int32(42))));
    }

    #[test]
    fn test_schema_evolution_add_field() {
        let old = MessageSchema::new("User")
            .add_field("name", FieldType::String, 1)
            .with_version(1);

        let new = MessageSchema::new("User")
            .add_field("name", FieldType::String, 1)
            .add_field("email", FieldType::String, 2)
            .with_version(2);

        let mut data = HashMap::new();
        data.insert(1, Value::String("alice".into()));

        let evolved = evolve_schema(&data, &old, &new).unwrap();
        assert_eq!(evolved.get(&1), Some(&Value::String("alice".into())));
        assert_eq!(evolved.get(&2), Some(&Value::String(String::new())));
    }

    #[test]
    fn test_schema_evolution_widen_type() {
        let old = MessageSchema::new("Metric")
            .add_field("value", FieldType::Int32, 1)
            .with_version(1);

        let new = MessageSchema::new("Metric")
            .add_field("value", FieldType::Float64, 1)
            .with_version(2);

        let mut data = HashMap::new();
        data.insert(1, Value::Integer(42));

        let evolved = evolve_schema(&data, &old, &new).unwrap();
        assert_eq!(evolved.get(&1), Some(&Value::Float(42.0)));
    }

    #[test]
    fn test_proto_encode_value_convenience() {
        let val = Value::Map(vec![
            (Value::String("x".into()), Value::Integer(10)),
            (Value::String("y".into()), Value::Float(2.5)),
        ]);
        let encoded = proto_encode_value(&val).unwrap();
        assert!(!encoded.is_empty());
        let mut dec = ProtoDecoder::new(&encoded);
        let fields = dec.decode_all().unwrap();
        assert!(!fields.is_empty());
    }

    #[test]
    fn test_msgpack_nested_structures() {
        let nested = Value::Array(vec![
            Value::Map(vec![
                (Value::String("id".into()), Value::Integer(1)),
                (Value::String("tags".into()), Value::Array(vec![
                    Value::String("a".into()),
                    Value::String("b".into()),
                ])),
            ]),
            Value::Map(vec![
                (Value::String("id".into()), Value::Integer(2)),
                (Value::String("tags".into()), Value::Array(vec![])),
            ]),
        ]);
        let encoded = msgpack_encode(&nested);
        let decoded = msgpack_decode(&encoded).unwrap();
        assert_eq!(nested, decoded);
    }
}
