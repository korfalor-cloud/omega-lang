use std::collections::HashMap;
use std::fmt;
use crate::errors::OmegaResult;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OmegaType {
    // Primitive types
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Float32,
    Float64,
    Bool,
    Char,
    String,
    None,
    Never,

    // Compound types
    Array(Box<OmegaType>, Option<usize>),
    Tuple(Vec<OmegaType>),
    Map(Box<OmegaType>, Box<OmegaType>),
    Set(Box<OmegaType>),
    Optional(Box<OmegaType>),
    Result(Box<OmegaType>, Box<OmegaType>),

    // Function type
    Function {
        params: Vec<OmegaType>,
        return_type: Box<OmegaType>,
        is_async: bool,
    },

    // Reference types
    Reference {
        mutable: bool,
        inner: Box<OmegaType>,
    },

    // User-defined types
    Struct(String, Vec<(String, OmegaType)>),
    Enum(String, HashMap<String, EnumVariantType>),
    Trait(String),
    Class(String),

    // Generic type
    Generic(String, Vec<OmegaType>),

    // Type parameter
    TypeParam(String),

    // Union type
    Union(Vec<OmegaType>),

    // Iterator type
    Iterator(Box<OmegaType>),

    // Future type
    Future(Box<OmegaType>),

    // Any type (top type)
    Any,

    // Self type
    SelfType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EnumVariantType {
    Unit,
    Tuple(Vec<OmegaType>),
    Struct(Vec<(String, OmegaType)>),
}

impl OmegaType {
    pub fn is_numeric(&self) -> bool {
        matches!(self,
            OmegaType::Int8 | OmegaType::Int16 | OmegaType::Int32 | OmegaType::Int64 | OmegaType::Int128 |
            OmegaType::UInt8 | OmegaType::UInt16 | OmegaType::UInt32 | OmegaType::UInt64 | OmegaType::UInt128 |
            OmegaType::Float32 | OmegaType::Float64
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(self,
            OmegaType::Int8 | OmegaType::Int16 | OmegaType::Int32 | OmegaType::Int64 | OmegaType::Int128 |
            OmegaType::UInt8 | OmegaType::UInt16 | OmegaType::UInt32 | OmegaType::UInt64 | OmegaType::UInt128
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, OmegaType::Float32 | OmegaType::Float64)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self,
            OmegaType::Int8 | OmegaType::Int16 | OmegaType::Int32 | OmegaType::Int64 | OmegaType::Int128
        )
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(self,
            OmegaType::UInt8 | OmegaType::UInt16 | OmegaType::UInt32 | OmegaType::UInt64 | OmegaType::UInt128
        )
    }

    pub fn is_primitive(&self) -> bool {
        matches!(self,
            OmegaType::Int8 | OmegaType::Int16 | OmegaType::Int32 | OmegaType::Int64 | OmegaType::Int128 |
            OmegaType::UInt8 | OmegaType::UInt16 | OmegaType::UInt32 | OmegaType::UInt64 | OmegaType::UInt128 |
            OmegaType::Float32 | OmegaType::Float64 | OmegaType::Bool | OmegaType::Char
        )
    }

    pub fn is_copyable(&self) -> bool {
        self.is_primitive() || matches!(self, OmegaType::Bool | OmegaType::Char)
    }

    pub fn is_collection(&self) -> bool {
        matches!(self,
            OmegaType::Array(_, _) | OmegaType::Tuple(_) |
            OmegaType::Map(_, _) | OmegaType::Set(_)
        )
    }

    pub fn size_hint(&self) -> usize {
        match self {
            OmegaType::Int8 | OmegaType::UInt8 | OmegaType::Bool | OmegaType::Char => 1,
            OmegaType::Int16 | OmegaType::UInt16 => 2,
            OmegaType::Int32 | OmegaType::UInt32 | OmegaType::Float32 => 4,
            OmegaType::Int64 | OmegaType::UInt64 | OmegaType::Float64 => 8,
            OmegaType::Int128 | OmegaType::UInt128 => 16,
            OmegaType::Array(inner, Some(size)) => inner.size_hint() * size,
            OmegaType::Array(inner, None) => inner.size_hint(),
            OmegaType::Tuple(types) => types.iter().map(|t| t.size_hint()).sum(),
            OmegaType::Reference { .. } | OmegaType::Function { .. } => 8, // pointer size
            _ => 16, // default
        }
    }

    pub fn unify(&self, other: &OmegaType) -> Option<OmegaType> {
        if self == other {
            return Some(self.clone());
        }

        match (self, other) {
            (OmegaType::Any, t) | (t, OmegaType::Any) => Some(t.clone()),
            (OmegaType::Optional(inner), &OmegaType::None) => Some(OmegaType::Optional(inner.clone())),
            (&OmegaType::None, OmegaType::Optional(inner)) => Some(OmegaType::Optional(inner.clone())),
            (OmegaType::Optional(a), OmegaType::Optional(b)) => {
                a.unify(b).map(|inner| OmegaType::Optional(Box::new(inner)))
            }
            (OmegaType::Union(types), other) | (other, OmegaType::Union(types)) => {
                if types.iter().any(|t| t == other) {
                    Some(other.clone())
                } else {
                    let mut new_types = types.clone();
                    new_types.push(other.clone());
                    Some(OmegaType::Union(new_types))
                }
            }
            (a, b) if a.is_numeric() && b.is_numeric() => {
                Some(promote_numeric(a, b))
            }
            _ => None,
        }
    }

    pub fn is_assignable_from(&self, other: &OmegaType) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (OmegaType::Any, _) => true,
            (_, OmegaType::Never) => true,
            (OmegaType::Optional(_), &OmegaType::None) => true,
            (OmegaType::Optional(inner), other) => inner.is_assignable_from(other),
            (a, b) if a.is_numeric() && b.is_numeric() => true,
            _ => false,
        }
    }

    pub fn default_value(&self) -> String {
        match self {
            OmegaType::Int8 | OmegaType::Int16 | OmegaType::Int32 | OmegaType::Int64 | OmegaType::Int128 |
            OmegaType::UInt8 | OmegaType::UInt16 | OmegaType::UInt32 | OmegaType::UInt64 | OmegaType::UInt128 => "0".to_string(),
            OmegaType::Float32 | OmegaType::Float64 => "0.0".to_string(),
            OmegaType::Bool => "false".to_string(),
            OmegaType::Char => "'\0'".to_string(),
            OmegaType::String => "\"\"".to_string(),
            OmegaType::None => "none".to_string(),
            OmegaType::Array(_, _) => "[]".to_string(),
            OmegaType::Tuple(_) => "()".to_string(),
            OmegaType::Map(_, _) => "{}".to_string(),
            OmegaType::Set(_) => "{}".to_string(),
            OmegaType::Optional(_) => "none".to_string(),
            _ => "default".to_string(),
        }
    }
}

impl fmt::Display for OmegaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OmegaType::Int8 => write!(f, "i8"),
            OmegaType::Int16 => write!(f, "i16"),
            OmegaType::Int32 => write!(f, "i32"),
            OmegaType::Int64 => write!(f, "i64"),
            OmegaType::Int128 => write!(f, "i128"),
            OmegaType::UInt8 => write!(f, "u8"),
            OmegaType::UInt16 => write!(f, "u16"),
            OmegaType::UInt32 => write!(f, "u32"),
            OmegaType::UInt64 => write!(f, "u64"),
            OmegaType::UInt128 => write!(f, "u128"),
            OmegaType::Float32 => write!(f, "f32"),
            OmegaType::Float64 => write!(f, "f64"),
            OmegaType::Bool => write!(f, "bool"),
            OmegaType::Char => write!(f, "char"),
            OmegaType::String => write!(f, "String"),
            OmegaType::None => write!(f, "None"),
            OmegaType::Never => write!(f, "never"),
            OmegaType::Array(inner, Some(size)) => write!(f, "[{}; {}]", inner, size),
            OmegaType::Array(inner, None) => write!(f, "[{}]", inner),
            OmegaType::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            OmegaType::Map(k, v) => write!(f, "Map<{}, {}>", k, v),
            OmegaType::Set(inner) => write!(f, "Set<{}>", inner),
            OmegaType::Optional(inner) => write!(f, "{}?", inner),
            OmegaType::Result(ok, err) => write!(f, "Result<{}, {}>", ok, err),
            OmegaType::Function { params, return_type, is_async } => {
                if *is_async { write!(f, "async ")?; }
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", return_type)
            }
            OmegaType::Reference { mutable, inner } => {
                if *mutable { write!(f, "&mut {}", inner) } else { write!(f, "&{}", inner) }
            }
            OmegaType::Struct(name, _) => write!(f, "{}", name),
            OmegaType::Enum(name, _) => write!(f, "{}", name),
            OmegaType::Trait(name) => write!(f, "dyn {}", name),
            OmegaType::Class(name) => write!(f, "{}", name),
            OmegaType::Generic(name, args) => {
                write!(f, "{}<", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ">")
            }
            OmegaType::TypeParam(name) => write!(f, "{}", name),
            OmegaType::Union(types) => {
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{}", t)?;
                }
                Ok(())
            }
            OmegaType::Iterator(inner) => write!(f, "Iterator<{}>", inner),
            OmegaType::Future(inner) => write!(f, "Future<{}>", inner),
            OmegaType::Any => write!(f, "Any"),
            OmegaType::SelfType => write!(f, "Self"),
        }
    }
}

fn promote_numeric(a: &OmegaType, b: &OmegaType) -> OmegaType {
    match (a, b) {
        (OmegaType::Float64, _) | (_, OmegaType::Float64) => OmegaType::Float64,
        (OmegaType::Float32, _) | (_, OmegaType::Float32) => OmegaType::Float32,
        (OmegaType::Int128, _) | (_, OmegaType::Int128) => OmegaType::Int128,
        (OmegaType::Int64, _) | (_, OmegaType::Int64) => OmegaType::Int64,
        (OmegaType::Int32, _) | (_, OmegaType::Int32) => OmegaType::Int32,
        (OmegaType::Int16, _) | (_, OmegaType::Int16) => OmegaType::Int16,
        _ => OmegaType::Int8,
    }
}

pub struct TypeRegistry {
    types: HashMap<String, OmegaType>,
    type_params: HashMap<String, Vec<String>>,
}

impl TypeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            types: HashMap::new(),
            type_params: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        self.types.insert("i8".to_string(), OmegaType::Int8);
        self.types.insert("i16".to_string(), OmegaType::Int16);
        self.types.insert("i32".to_string(), OmegaType::Int32);
        self.types.insert("i64".to_string(), OmegaType::Int64);
        self.types.insert("i128".to_string(), OmegaType::Int128);
        self.types.insert("u8".to_string(), OmegaType::UInt8);
        self.types.insert("u16".to_string(), OmegaType::UInt16);
        self.types.insert("u32".to_string(), OmegaType::UInt32);
        self.types.insert("u64".to_string(), OmegaType::UInt64);
        self.types.insert("u128".to_string(), OmegaType::UInt128);
        self.types.insert("f32".to_string(), OmegaType::Float32);
        self.types.insert("f64".to_string(), OmegaType::Float64);
        self.types.insert("bool".to_string(), OmegaType::Bool);
        self.types.insert("char".to_string(), OmegaType::Char);
        self.types.insert("String".to_string(), OmegaType::String);
        self.types.insert("None".to_string(), OmegaType::None);
        self.types.insert("never".to_string(), OmegaType::Never);
        self.types.insert("Any".to_string(), OmegaType::Any);
    }

    pub fn register_type(&mut self, name: String, ty: OmegaType) {
        self.types.insert(name, ty);
    }

    pub fn get_type(&self, name: &str) -> Option<&OmegaType> {
        self.types.get(name)
    }

    pub fn register_struct(&mut self, name: &str, fields: Vec<(String, OmegaType)>) {
        self.types.insert(name.to_string(), OmegaType::Struct(name.to_string(), fields));
    }

    pub fn register_enum(&mut self, name: &str, variants: HashMap<String, EnumVariantType>) {
        self.types.insert(name.to_string(), OmegaType::Enum(name.to_string(), variants));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_properties() {
        assert!(OmegaType::Int32.is_numeric());
        assert!(OmegaType::Float64.is_float());
        assert!(OmegaType::Bool.is_primitive());
        assert!(OmegaType::Array(Box::new(OmegaType::Int32), None).is_collection());
    }

    #[test]
    fn test_type_unify() {
        let result = OmegaType::Int32.unify(&OmegaType::Int64);
        assert_eq!(result, Some(OmegaType::Int64));

        let result = OmegaType::Optional(Box::new(OmegaType::Int32)).unify(&OmegaType::None);
        assert_eq!(result, Some(OmegaType::Optional(Box::new(OmegaType::Int32))));
    }

    #[test]
    fn test_type_display() {
        assert_eq!(OmegaType::Int32.to_string(), "i32");
        assert_eq!(OmegaType::Optional(Box::new(OmegaType::String)).to_string(), "String?");
        assert_eq!(
            OmegaType::Function {
                params: vec![OmegaType::Int32, OmegaType::Bool],
                return_type: Box::new(OmegaType::String),
                is_async: false,
            }.to_string(),
            "fn(i32, bool) -> String"
        );
    }

    #[test]
    fn test_type_registry() {
        let mut registry = TypeRegistry::new();
        assert_eq!(registry.get_type("i32"), Some(&OmegaType::Int32));
        assert_eq!(registry.get_type("String"), Some(&OmegaType::String));
    }
}
