use std::collections::HashMap;
use crate::types::OmegaType;

#[derive(Debug, Clone)]
pub struct Scope {
    pub variables: HashMap<String, VariableInfo>,
    pub functions: HashMap<String, FunctionInfo>,
    pub types: HashMap<String, OmegaType>,
    pub parent: Option<Box<Scope>>,
    pub depth: usize,
    pub scope_type: ScopeType,
}

#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub ty: OmegaType,
    pub mutable: bool,
    pub initialized: bool,
    pub used: bool,
    pub depth: usize,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub params: Vec<(String, OmegaType)>,
    pub return_type: OmegaType,
    pub is_async: bool,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    Global,
    Function,
    Block,
    Loop,
    Module,
    Class,
    Lambda,
}

impl Scope {
    pub fn global() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
            parent: None,
            depth: 0,
            scope_type: ScopeType::Global,
        }
    }

    pub fn child(&self, scope_type: ScopeType) -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
            parent: None,
            depth: self.depth + 1,
            scope_type,
        }
    }

    pub fn define_variable(&mut self, name: String, ty: OmegaType, mutable: bool) {
        self.variables.insert(name.clone(), VariableInfo {
            name,
            ty,
            mutable,
            initialized: false,
            used: false,
            depth: self.depth,
        });
    }

    pub fn initialize_variable(&mut self, name: &str) {
        if let Some(var) = self.variables.get_mut(name) {
            var.initialized = true;
        }
    }

    pub fn use_variable(&mut self, name: &str) -> bool {
        if let Some(var) = self.variables.get_mut(name) {
            var.used = true;
            true
        } else if let Some(parent) = &mut self.parent {
            parent.use_variable(name)
        } else {
            false
        }
    }

    pub fn get_variable(&self, name: &str) -> Option<&VariableInfo> {
        self.variables.get(name).or_else(|| {
            self.parent.as_ref().and_then(|p| p.get_variable(name))
        })
    }

    pub fn get_variable_mut(&mut self, name: &str) -> Option<&mut VariableInfo> {
        if self.variables.contains_key(name) {
            self.variables.get_mut(name)
        } else {
            self.parent.as_mut().and_then(|p| p.get_variable_mut(name))
        }
    }

    pub fn define_function(&mut self, name: String, info: FunctionInfo) {
        self.functions.insert(name, info);
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionInfo> {
        self.functions.get(name).or_else(|| {
            self.parent.as_ref().and_then(|p| p.get_function(name))
        })
    }

    pub fn define_type(&mut self, name: String, ty: OmegaType) {
        self.types.insert(name, ty);
    }

    pub fn get_type(&self, name: &str) -> Option<&OmegaType> {
        self.types.get(name).or_else(|| {
            self.parent.as_ref().and_then(|p| p.get_type(name))
        })
    }

    pub fn is_in_loop(&self) -> bool {
        self.scope_type == ScopeType::Loop ||
        self.parent.as_ref().map_or(false, |p| p.is_in_loop())
    }

    pub fn is_in_function(&self) -> bool {
        self.scope_type == ScopeType::Function ||
        self.parent.as_ref().map_or(false, |p| p.is_in_function())
    }

    pub fn unused_variables(&self) -> Vec<&VariableInfo> {
        self.variables.values().filter(|v| !v.used && v.depth > 0).collect()
    }

    pub fn unresolved_names(&self) -> Vec<String> {
        let mut unresolved = Vec::new();
        for var in self.variables.values() {
            if !var.initialized {
                unresolved.push(var.name.clone());
            }
        }
        unresolved
    }
}
