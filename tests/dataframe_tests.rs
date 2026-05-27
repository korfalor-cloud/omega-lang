use omega_lang::stdlib::data::dataframe::{DataFrame, Value};

#[test]
fn test_dataframe_new() {
    let df = DataFrame::new();
    assert_eq!(df.len(), 0);
    assert!(df.is_empty());
}

#[test]
fn test_dataframe_add_column() {
    let mut df = DataFrame::new();
    df.add_column("name", vec![
        Value::String("Alice".to_string()),
        Value::String("Bob".to_string()),
    ]).unwrap();

    assert_eq!(df.len(), 2);
    assert_eq!(df.num_columns(), 1);
}

#[test]
fn test_dataframe_multiple_columns() {
    let mut df = DataFrame::new();
    df.add_column("name", vec![
        Value::String("Alice".to_string()),
        Value::String("Bob".to_string()),
    ]).unwrap();
    df.add_column("age", vec![
        Value::Int(30),
        Value::Int(25),
    ]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.columns(), &["name", "age"]);
}

#[test]
fn test_dataframe_get_column() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![Value::Int(1), Value::Int(2)]).unwrap();

    let col = df.get_column("x").unwrap();
    assert_eq!(col.len(), 2);
}

#[test]
fn test_dataframe_get_row() {
    let mut df = DataFrame::new();
    df.add_column("name", vec![
        Value::String("Alice".to_string()),
        Value::String("Bob".to_string()),
    ]).unwrap();
    df.add_column("age", vec![Value::Int(30), Value::Int(25)]).unwrap();

    let row = df.get_row(0).unwrap();
    assert_eq!(row.len(), 2);
}

#[test]
fn test_dataframe_get_value() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![Value::Int(10), Value::Int(20)]).unwrap();

    let val = df.get_value("x", 0).unwrap();
    assert!(matches!(val, Value::Int(10)));
}

#[test]
fn test_dataframe_head() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![
        Value::Int(1), Value::Int(2), Value::Int(3),
        Value::Int(4), Value::Int(5),
    ]).unwrap();

    let head = df.head(3);
    assert_eq!(head.len(), 3);
}

#[test]
fn test_dataframe_tail() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![
        Value::Int(1), Value::Int(2), Value::Int(3),
        Value::Int(4), Value::Int(5),
    ]).unwrap();

    let tail = df.tail(3);
    assert_eq!(tail.len(), 3);
}

#[test]
fn test_dataframe_select() {
    let mut df = DataFrame::new();
    df.add_column("a", vec![Value::Int(1)]).unwrap();
    df.add_column("b", vec![Value::Int(2)]).unwrap();
    df.add_column("c", vec![Value::Int(3)]).unwrap();

    let selected = df.select(&["a", "c"]).unwrap();
    assert_eq!(selected.num_columns(), 2);
    assert_eq!(selected.columns(), &["a", "c"]);
}

#[test]
fn test_dataframe_filter() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![
        Value::Int(1), Value::Int(2), Value::Int(3),
        Value::Int(4), Value::Int(5),
    ]).unwrap();

    let filtered = df.filter(|df, i| {
        if let Value::Int(n) = df.get_value("x", i).unwrap() {
            *n > 3
        } else {
            false
        }
    });

    assert_eq!(filtered.len(), 2);
}

#[test]
fn test_dataframe_sort() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![
        Value::Int(3), Value::Int(1), Value::Int(2),
    ]).unwrap();

    let sorted = df.sort_by("x", true).unwrap();
    if let Value::Int(n) = sorted.get_value("x", 0).unwrap() {
        assert_eq!(*n, 1);
    }
}

#[test]
fn test_dataframe_to_csv() {
    let mut df = DataFrame::new();
    df.add_column("name", vec![
        Value::String("Alice".to_string()),
        Value::String("Bob".to_string()),
    ]).unwrap();
    df.add_column("age", vec![Value::Int(30), Value::Int(25)]).unwrap();

    let csv = df.to_csv();
    assert!(csv.contains("name,age"));
    assert!(csv.contains("Alice,30"));
    assert!(csv.contains("Bob,25"));
}

#[test]
fn test_dataframe_from_csv() {
    let csv = "name,age\nAlice,30\nBob,25";
    let df = DataFrame::from_csv(csv).unwrap();

    assert_eq!(df.len(), 2);
    assert_eq!(df.num_columns(), 2);
}

#[test]
fn test_dataframe_to_json() {
    let mut df = DataFrame::new();
    df.add_column("x", vec![Value::Int(1), Value::Int(2)]).unwrap();

    let json = df.to_json();
    assert!(json.contains("\"x\": 1"));
    assert!(json.contains("\"x\": 2"));
}

#[test]
fn test_value_display() {
    assert_eq!(format!("{}", Value::Null), "null");
    assert_eq!(format!("{}", Value::Bool(true)), "true");
    assert_eq!(format!("{}", Value::Int(42)), "42");
    assert_eq!(format!("{}", Value::Float(3.14)), "3.14");
    assert_eq!(format!("{}", Value::String("hello".to_string())), "hello");
}

#[test]
fn test_dataframe_display() {
    let mut df = DataFrame::new();
    df.add_column("name", vec![
        Value::String("Alice".to_string()),
        Value::String("Bob".to_string()),
    ]).unwrap();
    df.add_column("age", vec![Value::Int(30), Value::Int(25)]).unwrap();

    let display = format!("{}", df);
    assert!(display.contains("name"));
    assert!(display.contains("age"));
    assert!(display.contains("Alice"));
    assert!(display.contains("30"));
}
