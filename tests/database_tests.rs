use omega_lang::stdlib::database::connection::{Connection, Value};
use omega_lang::stdlib::database::query_builder::QueryBuilder;
use omega_lang::stdlib::database::migrations::{MigrationRunner, SchemaBuilder};

#[test]
fn test_connection() {
    let mut conn = Connection::new("sqlite:test.db");
    assert!(!conn.is_connected());
    conn.connect().unwrap();
    assert!(conn.is_connected());
    conn.disconnect();
    assert!(!conn.is_connected());
}

#[test]
fn test_value_types() {
    let v = Value::Integer(42);
    assert_eq!(v.as_integer(), Some(42));
    assert_eq!(v.as_float(), Some(42.0));
    assert!(!v.is_null());

    let v = Value::Text("hello".to_string());
    assert_eq!(v.as_text(), Some("hello"));

    let v = Value::Boolean(true);
    assert_eq!(v.as_bool(), Some(true));

    let v = Value::Null;
    assert!(v.is_null());
}

#[test]
fn test_query_builder_select() {
    let sql = QueryBuilder::select("users").build();
    assert_eq!(sql, "SELECT * FROM users");
}

#[test]
fn test_query_builder_where() {
    let sql = QueryBuilder::select("users")
        .where_eq("name", "Alice")
        .where_gt("age", "18")
        .build();
    assert!(sql.contains("WHERE"));
    assert!(sql.contains("name = 'Alice'"));
}

#[test]
fn test_query_builder_join() {
    let sql = QueryBuilder::select("orders")
        .join("users", "orders.user_id = users.id")
        .build();
    assert!(sql.contains("INNER JOIN users ON orders.user_id = users.id"));
}

#[test]
fn test_query_builder_insert() {
    let sql = QueryBuilder::insert("users")
        .value("name", "Alice")
        .value("age", "30")
        .build();
    assert!(sql.contains("INSERT INTO users"));
    assert!(sql.contains("'Alice'"));
}

#[test]
fn test_query_builder_update() {
    let sql = QueryBuilder::update("users")
        .set("name", "Bob")
        .where_eq("id", "1")
        .build();
    assert!(sql.contains("UPDATE users SET"));
    assert!(sql.contains("WHERE"));
}

#[test]
fn test_query_builder_delete() {
    let sql = QueryBuilder::delete("users")
        .where_eq("id", "1")
        .build();
    assert!(sql.contains("DELETE FROM users WHERE"));
}

#[test]
fn test_query_builder_order_limit() {
    let sql = QueryBuilder::select("users")
        .order_by("name", true)
        .limit(10)
        .offset(20)
        .build();
    assert!(sql.contains("ORDER BY name ASC"));
    assert!(sql.contains("LIMIT 10"));
    assert!(sql.contains("OFFSET 20"));
}

#[test]
fn test_migration_runner() {
    let mut runner = MigrationRunner::new();
    runner.add_migration(1, "create_users", "CREATE TABLE users", "DROP TABLE users");
    runner.add_migration(2, "add_email", "ALTER TABLE users ADD email TEXT", "ALTER TABLE users DROP email");

    let result = runner.up(None);
    assert_eq!(result.len(), 2);
    assert_eq!(runner.current_version(), Some(2));
}

#[test]
fn test_migration_rollback() {
    let mut runner = MigrationRunner::new();
    runner.add_migration(1, "create_users", "CREATE TABLE users", "DROP TABLE users");
    runner.up(None);

    let result = runner.down(1);
    assert_eq!(result.len(), 1);
    assert!(runner.current_version().is_none());
}

#[test]
fn test_schema_builder() {
    let sql = SchemaBuilder::create_table("users")
        .id()
        .string("name")
        .string("email")
        .not_null("name")
        .unique("email")
        .timestamps()
        .build();

    assert!(sql.contains("CREATE TABLE users"));
    assert!(sql.contains("id INTEGER PRIMARY KEY"));
    assert!(sql.contains("NOT NULL"));
}

#[test]
fn test_schema_indexes() {
    let indexes = SchemaBuilder::create_table("posts")
        .id()
        .string("title")
        .index("idx_title", &["title"])
        .build_indexes();

    assert_eq!(indexes.len(), 1);
    assert!(indexes[0].contains("CREATE INDEX"));
}
