/// SQL query builder.

use std::fmt;

#[derive(Debug, Clone)]
pub struct QueryBuilder {
    table: String,
    query_type: QueryType,
    columns: Vec<String>,
    conditions: Vec<Condition>,
    order_by: Vec<OrderBy>,
    limit: Option<usize>,
    offset: Option<usize>,
    joins: Vec<Join>,
    group_by: Vec<String>,
    having: Option<String>,
    values: Vec<(String, String)>,
    set_clauses: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub column: String,
    pub operator: String,
    pub value: String,
    pub connector: String,
}

#[derive(Debug, Clone)]
pub struct OrderBy {
    pub column: String,
    pub ascending: bool,
}

#[derive(Debug, Clone)]
pub struct Join {
    pub join_type: String,
    pub table: String,
    pub on: String,
}

impl QueryBuilder {
    pub fn select(table: &str) -> Self {
        Self {
            table: table.to_string(),
            query_type: QueryType::Select,
            columns: vec!["*".to_string()],
            conditions: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            joins: Vec::new(),
            group_by: Vec::new(),
            having: None,
            values: Vec::new(),
            set_clauses: Vec::new(),
        }
    }

    pub fn insert(table: &str) -> Self {
        Self {
            table: table.to_string(),
            query_type: QueryType::Insert,
            columns: Vec::new(),
            conditions: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            joins: Vec::new(),
            group_by: Vec::new(),
            having: None,
            values: Vec::new(),
            set_clauses: Vec::new(),
        }
    }

    pub fn update(table: &str) -> Self {
        Self {
            table: table.to_string(),
            query_type: QueryType::Update,
            columns: Vec::new(),
            conditions: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            joins: Vec::new(),
            group_by: Vec::new(),
            having: None,
            values: Vec::new(),
            set_clauses: Vec::new(),
        }
    }

    pub fn delete(table: &str) -> Self {
        Self {
            table: table.to_string(),
            query_type: QueryType::Delete,
            columns: Vec::new(),
            conditions: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            joins: Vec::new(),
            group_by: Vec::new(),
            having: None,
            values: Vec::new(),
            set_clauses: Vec::new(),
        }
    }

    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = cols.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn column(mut self, col: &str) -> Self {
        self.columns.push(col.to_string());
        self
    }

    pub fn where_eq(mut self, column: &str, value: &str) -> Self {
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: "=".to_string(),
            value: format!("'{}'", value.replace('\'', "''")),
            connector: "AND".to_string(),
        });
        self
    }

    pub fn where_gt(mut self, column: &str, value: &str) -> Self {
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: ">".to_string(),
            value: value.to_string(),
            connector: "AND".to_string(),
        });
        self
    }

    pub fn where_lt(mut self, column: &str, value: &str) -> Self {
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: "<".to_string(),
            value: value.to_string(),
            connector: "AND".to_string(),
        });
        self
    }

    pub fn where_like(mut self, column: &str, pattern: &str) -> Self {
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: "LIKE".to_string(),
            value: format!("'{}'", pattern),
            connector: "AND".to_string(),
        });
        self
    }

    pub fn where_in(mut self, column: &str, values: &[&str]) -> Self {
        let vals = values.iter()
            .map(|v| format!("'{}'", v.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: "IN".to_string(),
            value: format!("({})", vals),
            connector: "AND".to_string(),
        });
        self
    }

    pub fn or_where_eq(mut self, column: &str, value: &str) -> Self {
        self.conditions.push(Condition {
            column: column.to_string(),
            operator: "=".to_string(),
            value: format!("'{}'", value.replace('\'', "''")),
            connector: "OR".to_string(),
        });
        self
    }

    pub fn join(mut self, table: &str, on: &str) -> Self {
        self.joins.push(Join {
            join_type: "INNER".to_string(),
            table: table.to_string(),
            on: on.to_string(),
        });
        self
    }

    pub fn left_join(mut self, table: &str, on: &str) -> Self {
        self.joins.push(Join {
            join_type: "LEFT".to_string(),
            table: table.to_string(),
            on: on.to_string(),
        });
        self
    }

    pub fn order_by(mut self, column: &str, ascending: bool) -> Self {
        self.order_by.push(OrderBy {
            column: column.to_string(),
            ascending,
        });
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    pub fn group_by(mut self, column: &str) -> Self {
        self.group_by.push(column.to_string());
        self
    }

    pub fn having(mut self, condition: &str) -> Self {
        self.having = Some(condition.to_string());
        self
    }

    pub fn value(mut self, column: &str, value: &str) -> Self {
        self.values.push((column.to_string(), format!("'{}'", value.replace('\'', "''"))));
        self
    }

    pub fn set(mut self, column: &str, value: &str) -> Self {
        self.set_clauses.push((column.to_string(), format!("'{}'", value.replace('\'', "''"))));
        self
    }

    pub fn set_raw(mut self, column: &str, value: &str) -> Self {
        self.set_clauses.push((column.to_string(), value.to_string()));
        self
    }

    pub fn build(&self) -> String {
        match self.query_type {
            QueryType::Select => self.build_select(),
            QueryType::Insert => self.build_insert(),
            QueryType::Update => self.build_update(),
            QueryType::Delete => self.build_delete(),
            QueryType::CreateTable => String::new(),
        }
    }

    fn build_select(&self) -> String {
        let mut sql = format!("SELECT {} FROM {}", self.columns.join(", "), self.table);

        for join in &self.joins {
            sql.push_str(&format!(" {} JOIN {} ON {}", join.join_type, join.table, join.on));
        }

        if !self.conditions.is_empty() {
            sql.push_str(" WHERE ");
            let parts: Vec<String> = self.conditions.iter().enumerate().map(|(i, c)| {
                if i == 0 {
                    format!("{} {} {}", c.column, c.operator, c.value)
                } else {
                    format!("{} {} {} {}", c.connector, c.column, c.operator, c.value)
                }
            }).collect();
            sql.push_str(&parts.join(" "));
        }

        if !self.group_by.is_empty() {
            sql.push_str(&format!(" GROUP BY {}", self.group_by.join(", ")));
        }

        if let Some(having) = &self.having {
            sql.push_str(&format!(" HAVING {}", having));
        }

        if !self.order_by.is_empty() {
            let parts: Vec<String> = self.order_by.iter().map(|o| {
                format!("{} {}", o.column, if o.ascending { "ASC" } else { "DESC" })
            }).collect();
            sql.push_str(&format!(" ORDER BY {}", parts.join(", ")));
        }

        if let Some(limit) = self.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = self.offset {
            sql.push_str(&format!(" OFFSET {}", offset));
        }

        sql
    }

    fn build_insert(&self) -> String {
        let cols: Vec<&str> = self.values.iter().map(|(c, _)| c.as_str()).collect();
        let vals: Vec<&str> = self.values.iter().map(|(_, v)| v.as_str()).collect();
        format!("INSERT INTO {} ({}) VALUES ({})", self.table, cols.join(", "), vals.join(", "))
    }

    fn build_update(&self) -> String {
        let mut sql = format!("UPDATE {} SET ", self.table);
        let parts: Vec<String> = self.set_clauses.iter()
            .map(|(col, val)| format!("{} = {}", col, val))
            .collect();
        sql.push_str(&parts.join(", "));

        if !self.conditions.is_empty() {
            sql.push_str(" WHERE ");
            let parts: Vec<String> = self.conditions.iter().enumerate().map(|(i, c)| {
                if i == 0 {
                    format!("{} {} {}", c.column, c.operator, c.value)
                } else {
                    format!("{} {} {} {}", c.connector, c.column, c.operator, c.value)
                }
            }).collect();
            sql.push_str(&parts.join(" "));
        }

        sql
    }

    fn build_delete(&self) -> String {
        let mut sql = format!("DELETE FROM {}", self.table);

        if !self.conditions.is_empty() {
            sql.push_str(" WHERE ");
            let parts: Vec<String> = self.conditions.iter().enumerate().map(|(i, c)| {
                if i == 0 {
                    format!("{} {} {}", c.column, c.operator, c.value)
                } else {
                    format!("{} {} {} {}", c.connector, c.column, c.operator, c.value)
                }
            }).collect();
            sql.push_str(&parts.join(" "));
        }

        sql
    }
}

impl fmt::Display for QueryBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_all() {
        let sql = QueryBuilder::select("users").build();
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn test_select_with_where() {
        let sql = QueryBuilder::select("users")
            .where_eq("name", "Alice")
            .where_gt("age", "18")
            .build();
        assert!(sql.contains("WHERE"));
        assert!(sql.contains("name = 'Alice'"));
    }

    #[test]
    fn test_select_with_join() {
        let sql = QueryBuilder::select("orders")
            .join("users", "orders.user_id = users.id")
            .build();
        assert!(sql.contains("INNER JOIN users ON orders.user_id = users.id"));
    }

    #[test]
    fn test_insert() {
        let sql = QueryBuilder::insert("users")
            .value("name", "Alice")
            .value("age", "30")
            .build();
        assert!(sql.contains("INSERT INTO users"));
        assert!(sql.contains("'Alice'"));
    }

    #[test]
    fn test_update() {
        let sql = QueryBuilder::update("users")
            .set("name", "Bob")
            .where_eq("id", "1")
            .build();
        assert!(sql.contains("UPDATE users SET"));
        assert!(sql.contains("WHERE"));
    }

    #[test]
    fn test_delete() {
        let sql = QueryBuilder::delete("users")
            .where_eq("id", "1")
            .build();
        assert!(sql.contains("DELETE FROM users WHERE"));
    }

    #[test]
    fn test_order_by() {
        let sql = QueryBuilder::select("users")
            .order_by("name", true)
            .order_by("age", false)
            .build();
        assert!(sql.contains("ORDER BY name ASC, age DESC"));
    }

    #[test]
    fn test_limit_offset() {
        let sql = QueryBuilder::select("users")
            .limit(10)
            .offset(20)
            .build();
        assert!(sql.contains("LIMIT 10"));
        assert!(sql.contains("OFFSET 20"));
    }

    #[test]
    fn test_group_by_having() {
        let sql = QueryBuilder::select("orders")
            .column("user_id")
            .group_by("user_id")
            .having("COUNT(*) > 5")
            .build();
        assert!(sql.contains("GROUP BY user_id"));
        assert!(sql.contains("HAVING COUNT(*) > 5"));
    }

    #[test]
    fn test_complex_query() {
        let sql = QueryBuilder::select("orders")
            .columns(&["orders.id", "users.name", "orders.total"])
            .join("users", "orders.user_id = users.id")
            .where_gt("orders.total", "100")
            .order_by("orders.total", false)
            .limit(10)
            .build();
        assert!(sql.contains("JOIN"));
        assert!(sql.contains("WHERE"));
        assert!(sql.contains("ORDER BY"));
        assert!(sql.contains("LIMIT"));
    }
}
