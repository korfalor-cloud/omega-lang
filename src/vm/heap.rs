use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};

#[derive(Debug, Clone)]
pub struct Object {
    pub type_name: String,
    pub fields: HashMap<String, super::stack::Value>,
    pub methods: HashMap<String, super::stack::FunctionValue>,
}

impl Object {
    pub fn new(type_name: String) -> Self {
        Self {
            type_name,
            fields: HashMap::new(),
            methods: HashMap::new(),
        }
    }

    pub fn get_field(&self, name: &str) -> Option<&super::stack::Value> {
        self.fields.get(name)
    }

    pub fn set_field(&mut self, name: String, value: super::stack::Value) {
        self.fields.insert(name, value);
    }

    pub fn get_method(&self, name: &str) -> Option<&super::stack::FunctionValue> {
        self.methods.get(name)
    }
}

#[derive(Debug)]
pub struct Heap {
    objects: Vec<HeapObject>,
    free_list: Vec<usize>,
    allocated: usize,
    next_gc: usize,
    gc_threshold: usize,
}

#[derive(Debug)]
struct HeapObject {
    value: super::stack::Value,
    marked: bool,
    alive: bool,
}

impl Heap {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            free_list: Vec::new(),
            allocated: 0,
            next_gc: 1024,
            gc_threshold: 1024,
        }
    }

    pub fn allocate(&mut self, value: super::stack::Value) -> usize {
        if let Some(index) = self.free_list.pop() {
            self.objects[index] = HeapObject {
                value,
                marked: false,
                alive: true,
            };
            self.allocated += 1;
            index
        } else {
            let index = self.objects.len();
            self.objects.push(HeapObject {
                value,
                marked: false,
                alive: true,
            });
            self.allocated += 1;
            index
        }
    }

    pub fn get(&self, index: usize) -> Option<&super::stack::Value> {
        self.objects.get(index).and_then(|obj| {
            if obj.alive {
                Some(&obj.value)
            } else {
                None
            }
        })
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut super::stack::Value> {
        self.objects.get_mut(index).and_then(|obj| {
            if obj.alive {
                Some(&mut obj.value)
            } else {
                None
            }
        })
    }

    pub fn free(&mut self, index: usize) {
        if let Some(obj) = self.objects.get_mut(index) {
            if obj.alive {
                obj.alive = false;
                self.free_list.push(index);
                self.allocated -= 1;
            }
        }
    }

    pub fn mark(&mut self, index: usize) {
        if let Some(obj) = self.objects.get_mut(index) {
            if !obj.marked && obj.alive {
                obj.marked = true;
                // Mark references within the value
                self.mark_value(&obj.value.clone());
            }
        }
    }

    fn mark_value(&mut self, value: &super::stack::Value) {
        match value {
            super::stack::Value::Array(elements) => {
                for elem in elements {
                    self.mark_value(elem);
                }
            }
            super::stack::Value::Map(entries) => {
                for (k, v) in entries {
                    self.mark_value(k);
                    self.mark_value(v);
                }
            }
            super::stack::Value::Tuple(elements) => {
                for elem in elements {
                    self.mark_value(elem);
                }
            }
            super::stack::Value::Object(obj) => {
                for (_, v) in &obj.fields {
                    self.mark_value(v);
                }
            }
            _ => {}
        }
    }

    pub fn sweep(&mut self) {
        for i in 0..self.objects.len() {
            if self.objects[i].alive && !self.objects[i].marked {
                self.objects[i].alive = false;
                self.free_list.push(i);
                self.allocated -= 1;
            }
            self.objects[i].marked = false;
        }
    }

    pub fn gc(&mut self, stack_roots: &[super::stack::Value]) {
        // Mark roots
        for root in stack_roots {
            self.mark_value(root);
        }
        // Sweep unreachable
        self.sweep();
    }

    pub fn should_gc(&self) -> bool {
        self.allocated >= self.next_gc
    }

    pub fn allocated_count(&self) -> usize {
        self.allocated
    }

    pub fn total_objects(&self) -> usize {
        self.objects.len()
    }

    pub fn compact(&mut self) {
        self.free_list.clear();
        self.objects.retain(|obj| obj.alive);
    }
}
