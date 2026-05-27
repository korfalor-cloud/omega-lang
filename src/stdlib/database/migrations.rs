/// Database migration system.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Migration {
    pub version: u32,
    pub name: String,
    pub up_sql: String,
    pub down_sql: String,
}

#[derive(Debug)]
pub struct MigrationRunner {
    migrations: Vec<Migration>,
    applied: Vec<u32>,
}

impl MigrationRunner {
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
            applied: Vec::new(),
        }
    }

    pub fn add_migration(&mut self, version: u32, name: &str, up: &str, down: &str) {
        self.migrations.push(Migration {
            version,
            name: name.to_string(),
            up_sql: up.to_string(),
            down_sql: down.to_string(),
        });
        self.migrations.sort_by_key(|m| m.version);
    }

    pub fn up(&mut self, target_version: Option<u32>) -> Vec<String> {
        let mut executed = Vec::new();
        let target = target_version.unwrap_or(u32::MAX);

        for migration in &self.migrations {
            if migration.version > target {
                break;
            }
            if !self.applied.contains(&migration.version) {
                executed.push(format!("Applying migration {}: {}", migration.version, migration.name));
                self.applied.push(migration.version);
            }
        }

        executed
    }

    pub fn down(&mut self, steps: usize) -> Vec<String> {
        let mut rolled_back = Vec::new();
        let mut to_rollback: Vec<u32> = self.applied.iter().rev().take(steps).cloned().collect();

        for version in to_rollback.drain(..) {
            if let Some(migration) = self.migrations.iter().find(|m| m.version == version) {
                rolled_back.push(format!("Rolling back migration {}: {}", migration.version, migration.name));
                self.applied.retain(|&v| v != version);
            }
        }

        rolled_back
    }

    pub fn status(&self) -> Vec<(u32, String, bool)> {
        self.migrations.iter().map(|m| {
            (m.version, m.name.clone(), self.applied.contains(&m.version))
        }).collect()
    }

    pub fn current_version(&self) -> Option<u32> {
        self.applied.iter().max().cloned()
    }

    pub fn pending_count(&self) -> usize {
        self.migrations.iter()
            .filter(|m| !self.applied.contains(&m.version))
            .count()
    }
}

/// Schema builder for migrations
pub struct SchemaBuilder {
    table: String,
    columns: Vec<ColumnDef>,
    indexes: Vec<IndexDef>,
}

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: String,
    pub nullable: bool,
    pub default: Option<String>,
    pub primary_key: bool,
    pub unique: bool,
    pub auto_increment: bool,
}

#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

impl SchemaBuilder {
    pub fn create_table(name: &str) -> Self {
        Self {
            table: name.to_string(),
            columns: Vec::new(),
            indexes: Vec::new(),
        }
    }

    pub fn id(mut self) -> Self {
        self.columns.push(ColumnDef {
            name: "id".to_string(),
            col_type: "INTEGER".to_string(),
            nullable: false,
            default: None,
            primary_key: true,
            unique: false,
            auto_increment: true,
        });
        self
    }

    pub fn string(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "VARCHAR(255)".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn text(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "TEXT".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn integer(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "INTEGER".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn float(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "REAL".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn boolean(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "BOOLEAN".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn timestamp(mut self, name: &str) -> Self {
        self.columns.push(ColumnDef {
            name: name.to_string(),
            col_type: "TIMESTAMP".to_string(),
            nullable: true,
            default: None,
            primary_key: false,
            unique: false,
            auto_increment: false,
        });
        self
    }

    pub fn not_null(mut self, name: &str) -> Self {
        if let Some(col) = self.columns.iter_mut().find(|c| c.name == name) {
            col.nullable = false;
        }
        self
    }

    pub fn default(mut self, name: &str, value: &str) -> Self {
        if let Some(col) = self.columns.iter_mut().find(|c| c.name == name) {
            col.default = Some(value.to_string());
        }
        self
    }

    pub fn unique(mut self, name: &str) -> Self {
        if let Some(col) = self.columns.iter_mut().find(|c| c.name == name) {
            col.unique = true;
        }
        self
    }

    pub fn index(mut self, name: &str, columns: &[&str]) -> Self {
        self.indexes.push(IndexDef {
            name: name.to_string(),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            unique: false,
        });
        self
    }

    pub fn unique_index(mut self, name: &str, columns: &[&str]) -> Self {
        self.indexes.push(IndexDef {
            name: name.to_string(),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            unique: true,
        });
        self
    }

    pub fn timestamps(self) -> Self {
        self.timestamp("created_at")
            .timestamp("updated_at")
            .default("created_at", "CURRENT_TIMESTAMP")
            .default("updated_at", "CURRENT_TIMESTAMP")
    }

    pub fn build(&self) -> String {
        let mut sql = format!("CREATE TABLE {} (\n", self.table);

        let col_defs: Vec<String> = self.columns.iter().map(|col| {
            let mut def = format!("  {} {}", col.name, col.col_type);
            if col.primary_key {
                def.push_str(" PRIMARY KEY");
            }
            if col.auto_increment {
                def.push_str(" AUTOINCREMENT");
            }
            if !col.nullable {
                def.push_str(" NOT NULL");
            }
            if col.unique {
                def.push_str(" UNIQUE");
            }
            if let Some(default) = &col.default {
                def.push_str(&format!(" DEFAULT {}", default));
            }
            def
        }).collect();

        sql.push_str(&col_defs.join(",\n"));
        sql.push_str("\n)");

        sql
    }

    pub fn build_indexes(&self) -> Vec<String> {
        self.indexes.iter().map(|idx| {
            let unique = if idx.unique { "UNIQUE " } else { "" };
            format!("CREATE {}INDEX {} ON {} ({})",
                unique, idx.name, self.table, idx.columns.join(", "))
        }).collect()
    }
}

pub fn drop_table(name: &str) -> String {
    format!("DROP TABLE IF EXISTS {}", name)
}

pub fn rename_table(old: &str, new: &str) -> String {
    format!("ALTER TABLE {} RENAME TO {}", old, new)
}

pub fn add_column(table: &str, column: &ColumnDef) -> String {
    let mut sql = format!("ALTER TABLE {} ADD COLUMN {} {}", table, column.name, column.col_type);
    if !column.nullable {
        sql.push_str(" NOT NULL");
    }
    if let Some(default) = &column.default {
        sql.push_str(&format!(" DEFAULT {}", default));
    }
    sql
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_runner() {
        let mut runner = MigrationRunner::new();
        runner.add_migration(1, "create_users", "CREATE TABLE users (id INTEGER)", "DROP TABLE users");
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

    #[test]
    fn test_migration_status() {
        let mut runner = MigrationRunner::new();
        runner.add_migration(1, "first", "SELECT 1", "SELECT 1");
        runner.add_migration(2, "second", "SELECT 1", "SELECT 1");
        runner.up(Some(1));

        let status = runner.status();
        assert_eq!(status.len(), 2);
        assert!(status[0].2); // applied
        assert!(!status[1].2); // not applied
    }
}
