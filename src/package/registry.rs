use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};

pub struct PackageRegistry {
    packages: HashMap<String, Vec<PackageVersion>>,
}

#[derive(Debug, Clone)]
pub struct PackageVersion {
    pub name: String,
    pub version: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub download_url: Option<String>,
}

impl PackageRegistry {
    pub fn new() -> Self {
        Self {
            packages: HashMap::new(),
        }
    }

    pub fn register(&mut self, package: PackageVersion) {
        self.packages
            .entry(package.name.clone())
            .or_insert_with(Vec::new)
            .push(package);
    }

    pub fn get(&self, name: &str, version: Option<&str>) -> Option<&PackageVersion> {
        self.packages.get(name).and_then(|versions| {
            if let Some(v) = version {
                versions.iter().find(|p| p.version == v)
            } else {
                versions.last()
            }
        })
    }

    pub fn list_versions(&self, name: &str) -> Vec<&str> {
        self.packages.get(name)
            .map(|versions| versions.iter().map(|p| p.version.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn search(&self, query: &str) -> Vec<&PackageVersion> {
        let query_lower = query.to_lowercase();
        self.packages.values()
            .flat_map(|versions| versions.last())
            .filter(|p| {
                p.name.to_lowercase().contains(&query_lower) ||
                p.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn all_packages(&self) -> Vec<&PackageVersion> {
        self.packages.values()
            .filter_map(|versions| versions.last())
            .collect()
    }
}
