use omega_lang::types::type_system::{OmegaType, TypeRegistry};

#[test]
fn test_integer_types() {
    assert_eq!(OmegaType::Int8.to_string(), "i8");
    assert_eq!(OmegaType::Int16.to_string(), "i16");
    assert_eq!(OmegaType::Int32.to_string(), "i32");
    assert_eq!(OmegaType::Int64.to_string(), "i64");
    assert_eq!(OmegaType::Int128.to_string(), "i128");
}

#[test]
fn test_unsigned_types() {
    assert_eq!(OmegaType::UInt8.to_string(), "u8");
    assert_eq!(OmegaType::UInt16.to_string(), "u16");
    assert_eq!(OmegaType::UInt32.to_string(), "u32");
    assert_eq!(OmegaType::UInt64.to_string(), "u64");
}

#[test]
fn test_float_types() {
    assert_eq!(OmegaType::Float32.to_string(), "f32");
    assert_eq!(OmegaType::Float64.to_string(), "f64");
}

#[test]
fn test_bool_type() {
    assert_eq!(OmegaType::Bool.to_string(), "bool");
}

#[test]
fn test_string_type() {
    assert_eq!(OmegaType::String.to_string(), "String");
}

#[test]
fn test_char_type() {
    assert_eq!(OmegaType::Char.to_string(), "char");
}

#[test]
fn test_array_type() {
    let arr = OmegaType::Array(Box::new(OmegaType::Int64));
    assert_eq!(arr.to_string(), "[i64]");
}

#[test]
fn test_map_type() {
    let map = OmegaType::Map(Box::new(OmegaType::String), Box::new(OmegaType::Int64));
    assert_eq!(map.to_string(), "Map<String, i64>");
}

#[test]
fn test_tuple_type() {
    let tuple = OmegaType::Tuple(vec![
        OmegaType::Int64,
        OmegaType::String,
        OmegaType::Bool,
    ]);
    assert_eq!(tuple.to_string(), "(i64, String, bool)");
}

#[test]
fn test_optional_type() {
    let opt = OmegaType::Optional(Box::new(OmegaType::Int64));
    assert_eq!(opt.to_string(), "i64?");
}

#[test]
fn test_function_type() {
    let func = OmegaType::Function {
        params: vec![OmegaType::Int64, OmegaType::Int64],
        return_type: Box::new(OmegaType::Int64),
    };
    assert_eq!(func.to_string(), "fn(i64, i64) -> i64");
}

#[test]
fn test_void_type() {
    assert_eq!(OmegaType::Void.to_string(), "void");
}

#[test]
fn test_none_type() {
    assert_eq!(OmegaType::None.to_string(), "None");
}

#[test]
fn test_any_type() {
    assert_eq!(OmegaType::Any.to_string(), "Any");
}

#[test]
fn test_numeric_compatibility() {
    assert!(OmegaType::Int8.is_compatible_with(&OmegaType::Int16));
    assert!(OmegaType::Int16.is_compatible_with(&OmegaType::Int32));
    assert!(OmegaType::Int32.is_compatible_with(&OmegaType::Int64));
    assert!(OmegaType::Float32.is_compatible_with(&OmegaType::Float64));
}

#[test]
fn test_same_type_compatibility() {
    assert!(OmegaType::Int64.is_compatible_with(&OmegaType::Int64));
    assert!(OmegaType::String.is_compatible_with(&OmegaType::String));
    assert!(OmegaType::Bool.is_compatible_with(&OmegaType::Bool));
}

#[test]
fn test_incompatible_types() {
    assert!(!OmegaType::Int64.is_compatible_with(&OmegaType::String));
    assert!(!OmegaType::Bool.is_compatible_with(&OmegaType::Float64));
}

#[test]
fn test_any_compatibility() {
    assert!(OmegaType::Any.is_compatible_with(&OmegaType::Int64));
    assert!(OmegaType::Any.is_compatible_with(&OmegaType::String));
    assert!(OmegaType::Int64.is_compatible_with(&OmegaType::Any));
}

#[test]
fn test_array_element_compatibility() {
    let arr1 = OmegaType::Array(Box::new(OmegaType::Int64));
    let arr2 = OmegaType::Array(Box::new(OmegaType::Int32));
    assert!(arr1.is_compatible_with(&arr2));
}

#[test]
fn test_map_compatibility() {
    let map1 = OmegaType::Map(Box::new(OmegaType::String), Box::new(OmegaType::Int64));
    let map2 = OmegaType::Map(Box::new(OmegaType::String), Box::new(OmegaType::Int64));
    assert!(map1.is_compatible_with(&map2));
}

#[test]
fn test_tuple_compatibility() {
    let tuple1 = OmegaType::Tuple(vec![OmegaType::Int64, OmegaType::String]);
    let tuple2 = OmegaType::Tuple(vec![OmegaType::Int64, OmegaType::String]);
    assert!(tuple1.is_compatible_with(&tuple2));
}

#[test]
fn test_optional_compatibility() {
    let opt = OmegaType::Optional(Box::new(OmegaType::Int64));
    assert!(opt.is_compatible_with(&OmegaType::None));
    assert!(opt.is_compatible_with(&OmegaType::Int64));
}

#[test]
fn test_type_registry() {
    let mut registry = TypeRegistry::new();
    registry.register_type("Point", OmegaType::Struct("Point".to_string()));
    assert!(registry.get_type("Point").is_some());
}

#[test]
fn test_type_size() {
    assert_eq!(OmegaType::Int8.size_hint(), Some(1));
    assert_eq!(OmegaType::Int16.size_hint(), Some(2));
    assert_eq!(OmegaType::Int32.size_hint(), Some(4));
    assert_eq!(OmegaType::Int64.size_hint(), Some(8));
    assert_eq!(OmegaType::Float32.size_hint(), Some(4));
    assert_eq!(OmegaType::Float64.size_hint(), Some(8));
    assert_eq!(OmegaType::Bool.size_hint(), Some(1));
    assert_eq!(OmegaType::Char.size_hint(), Some(4));
}

#[test]
fn test_type_is_numeric() {
    assert!(OmegaType::Int8.is_numeric());
    assert!(OmegaType::Int64.is_numeric());
    assert!(OmegaType::UInt64.is_numeric());
    assert!(OmegaType::Float32.is_numeric());
    assert!(OmegaType::Float64.is_numeric());
    assert!(!OmegaType::String.is_numeric());
    assert!(!OmegaType::Bool.is_numeric());
}

#[test]
fn test_type_is_integer() {
    assert!(OmegaType::Int8.is_integer());
    assert!(OmegaType::Int64.is_integer());
    assert!(OmegaType::UInt64.is_integer());
    assert!(!OmegaType::Float64.is_integer());
    assert!(!OmegaType::String.is_integer());
}

#[test]
fn test_type_is_float() {
    assert!(OmegaType::Float32.is_float());
    assert!(OmegaType::Float64.is_float());
    assert!(!OmegaType::Int64.is_float());
}

#[test]
fn test_type_is_signed() {
    assert!(OmegaType::Int8.is_signed());
    assert!(OmegaType::Int64.is_signed());
    assert!(!OmegaType::UInt64.is_signed());
}

#[test]
fn test_type_is_unsigned() {
    assert!(OmegaType::UInt8.is_unsigned());
    assert!(OmegaType::UInt64.is_unsigned());
    assert!(!OmegaType::Int64.is_unsigned());
}

#[test]
fn test_unify_same_types() {
    let result = OmegaType::unify(&OmegaType::Int64, &OmegaType::Int64);
    assert_eq!(result, Some(OmegaType::Int64));
}

#[test]
fn test_unify_numeric_types() {
    let result = OmegaType::unify(&OmegaType::Int32, &OmegaType::Int64);
    assert_eq!(result, Some(OmegaType::Int64));
}

#[test]
fn test_unify_float_types() {
    let result = OmegaType::unify(&OmegaType::Float32, &OmegaType::Float64);
    assert_eq!(result, Some(OmegaType::Float64));
}

#[test]
fn test_unify_int_float() {
    let result = OmegaType::unify(&OmegaType::Int64, &OmegaType::Float64);
    assert_eq!(result, Some(OmegaType::Float64));
}

#[test]
fn test_unify_incompatible() {
    let result = OmegaType::unify(&OmegaType::Int64, &OmegaType::String);
    assert_eq!(result, None);
}

#[test]
fn test_unify_arrays() {
    let arr1 = OmegaType::Array(Box::new(OmegaType::Int32));
    let arr2 = OmegaType::Array(Box::new(OmegaType::Int64));
    let result = OmegaType::unify(&arr1, &arr2);
    assert_eq!(result, Some(OmegaType::Array(Box::new(OmegaType::Int64))));
}

#[test]
fn test_unify_tuples() {
    let t1 = OmegaType::Tuple(vec![OmegaType::Int32, OmegaType::Float32]);
    let t2 = OmegaType::Tuple(vec![OmegaType::Int64, OmegaType::Float64]);
    let result = OmegaType::unify(&t1, &t2);
    assert_eq!(
        result,
        Some(OmegaType::Tuple(vec![OmegaType::Int64, OmegaType::Float64]))
    );
}

#[test]
fn test_numeric_promotion() {
    assert_eq!(
        OmegaType::promote_numeric(&OmegaType::Int8, &OmegaType::Int64),
        OmegaType::Int64
    );
    assert_eq!(
        OmegaType::promote_numeric(&OmegaType::Int32, &OmegaType::Float64),
        OmegaType::Float64
    );
    assert_eq!(
        OmegaType::promote_numeric(&OmegaType::Float32, &OmegaType::Float64),
        OmegaType::Float64
    );
}
