use std::collections::{HashMap, HashSet};
use crate::errors::{OmegaError, OmegaResult};
use super::manifest::{PackageManifest, Dependency};

pub struct DependencyResolver {
    resolved: HashMap<String, String>,
    visiting: HashSet<String>,
}

impl DependencyResolver {
    pub fn new() -> Self {
        Self {
            resolved: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    pub fn resolve(&mut self, manifest: &PackageManifest) -> OmegaResult<HashMap<String, String>> {
        self.resolved.clear();
        self.visiting.clear();

        for (name, dep) in manifest.dependencies() {
            self.resolve_dep(name, dep)?;
        }

        Ok(self.resolved.clone())
    }

    fn resolve_dep(&mut self, name: &str, dep: &Dependency) -> OmegaResult<()> {
        if self.resolved.contains_key(name) {
            return Ok(());
        }

        if self.visiting.contains(name) {
            return Err(OmegaError::PackageError {
                message: format!("Circular dependency detected: {}", name),
            });
        }

        self.visiting.insert(name.to_string());

        let version = match dep {
            Dependency::Version(v) => v.clone(),
            Dependency::Detailed(d) => d.version.clone().unwrap_or_else(|| "*".to_string()),
        };

        self.resolved.insert(name.to_string(), version);
        self.visiting.remove(name);

        Ok(())
    }

    pub fn check_compatibility(&self, manifest: &PackageManifest) -> OmegaResult<()> {
        for (name, dep) in manifest.dependencies() {
            let required_version = match dep {
                Dependency::Version(v) => v.clone(),
                Dependency::Detailed(d) => d.version.clone().unwrap_or_else(|| "*".to_string()),
            };

            if let Some(resolved) = self.resolved.get(name) {
                if !self.version_compatible(resolved, &required_version) {
                    return Err(OmegaError::PackageError {
                        message: format!(
                            "Version conflict for {}: required {}, resolved {}",
                            name, required_version, resolved
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn version_compatible(&self, resolved: &str, required: &str) -> bool {
        if required == "*" {
            return true;
        }
        // Simple version comparison - in production use semver
        resolved == required
    }
}
