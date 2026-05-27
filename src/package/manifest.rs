use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub package: PackageInfo,
    pub dependencies: Option<HashMap<String, Dependency>>,
    pub dev_dependencies: Option<HashMap<String, Dependency>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub authors: Option<Vec<String>>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub edition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    Version(String),
    Detailed(DetailedDependency),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedDependency {
    pub version: Option<String>,
    pub path: Option<String>,
    pub git: Option<String>,
    pub branch: Option<String>,
    pub tag: Option<String>,
    pub rev: Option<String>,
    pub features: Option<Vec<String>>,
    pub optional: Option<bool>,
}

impl PackageManifest {
    pub fn from_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn to_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn name(&self) -> &str {
        &self.package.name
    }

    pub fn version(&self) -> &str {
        &self.package.version
    }

    pub fn dependencies(&self) -> &HashMap<String, Dependency> {
        self.dependencies.as_ref().unwrap_or(&HashMap::new())
    }

    pub fn dev_dependencies(&self) -> &HashMap<String, Dependency> {
        self.dev_dependencies.as_ref().unwrap_or(&HashMap::new())
    }
}

impl Default for PackageManifest {
    fn default() -> Self {
        Self {
            package: PackageInfo {
                name: String::new(),
                version: "0.1.0".to_string(),
                description: None,
                authors: None,
                license: None,
                repository: None,
                homepage: None,
                keywords: None,
                edition: Some("2024".to_string()),
            },
            dependencies: None,
            dev_dependencies: None,
        }
    }
}
