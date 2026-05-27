use crate::ast::*;
use crate::errors::{Diagnostic, DiagnosticBag, OmegaError, OmegaResult, Span};
use crate::types::{OmegaType, TypeRegistry};
use super::scope::{Scope, ScopeType, VariableInfo, FunctionInfo};

pub struct SemanticAnalyzer {
    scope: Scope,
    type_registry: TypeRegistry,
    diagnostics: DiagnosticBag,
    current_return_type: Option<OmegaType>,
    in_loop: bool,
    in_function: bool,
}

impl SemanticAnalyzer {
    pub fn new(source: &str) -> Self {
        Self {
            scope: Scope::global(),
            type_registry: TypeRegistry::new(),
            diagnostics: DiagnosticBag::new(source, "<input>"),
            current_return_type: None,
            in_loop: false,
            in_function: false,
        }
    }

    pub fn analyze(&mut self, ast: &AstNode) -> OmegaResult<()> {
        self.visit_node(ast)?;
        Ok(())
    }

    fn visit_node(&mut self, node: &AstNode) -> OmegaResult<OmegaType> {
        match node {
            AstNode::Program(stmts) => {
                let mut last_type = OmegaType::None;
                for stmt in stmts {
                    last_type = self.visit_node(stmt)?;
                }
                Ok(last_type)
            }
            AstNode::Block(stmts) => {
                self.scope = self.scope.child(ScopeType::Block);
                let mut last_type = OmegaType::None;
                for stmt in stmts {
                    last_type = self.visit_node(stmt)?;
                }
                self.check_unused();
                self.scope = *self.scope.parent.take().unwrap_or(Box::new(Scope::global()));
                Ok(last_type)
            }
            AstNode::IntegerLiteral(_) => Ok(OmegaType::Int64),
            AstNode::FloatLiteral(_) => Ok(OmegaType::Float64),
            AstNode::StringLiteral(_) => Ok(OmegaType::String),
            AstNode::BoolLiteral(_) => Ok(OmegaType::Bool),
            AstNode::CharLiteral(_) => Ok(OmegaType::Char),
            AstNode::NoneLiteral => Ok(OmegaType::None),
            AstNode::BigIntLiteral(_) => Ok(OmegaType::Int128),
            AstNode::Identifier(name) => {
                if let Some(var) = self.scope.get_variable(name) {
                    if !var.initialized {
                        return Err(OmegaError::NameError {
                            name: name.clone(),
                            span: None,
                        });
                    }
                    self.scope.use_variable(name);
                    Ok(var.ty.clone())
                } else if let Some(func) = self.scope.get_function(name) {
                    Ok(OmegaType::Function {
                        params: func.params.iter().map(|(_, t)| t.clone()).collect(),
                        return_type: Box::new(func.return_type.clone()),
                        is_async: func.is_async,
                    })
                } else {
                    Err(OmegaError::NameError {
                        name: name.clone(),
                        span: None,
                    })
                }
            }
            AstNode::LetBinding { name, mutable, type_annotation, value } => {
                let ty = if let Some(ann) = type_annotation {
                    self.resolve_type(ann)?
                } else if let Some(val) = value {
                    self.visit_node(val)?
                } else {
                    return Err(OmegaError::TypeError {
                        message: "Variable must have type annotation or initializer".to_string(),
                        span: None,
                    });
                };

                if let Some(val) = value {
                    let val_ty = self.visit_node(val)?;
                    if !ty.is_assignable_from(&val_ty) {
                        return Err(OmegaError::TypeError {
                            message: format!("Cannot assign {} to {}", val_ty, ty),
                            span: None,
                        });
                    }
                }

                self.scope.define_variable(name.clone(), ty.clone(), *mutable);
                self.scope.initialize_variable(name);
                Ok(ty)
            }
            AstNode::ConstBinding { name, type_annotation, value } => {
                let ty = if let Some(ann) = type_annotation {
                    self.resolve_type(ann)?
                } else {
                    self.visit_node(value)?
                };

                let val_ty = self.visit_node(value)?;
                if !ty.is_assignable_from(&val_ty) {
                    return Err(OmegaError::TypeError {
                        message: format!("Cannot assign {} to {}", val_ty, ty),
                        span: None,
                    });
                }

                self.scope.define_variable(name.clone(), ty.clone(), false);
                self.scope.initialize_variable(name);
                Ok(ty)
            }
            AstNode::BinaryOp { op, left, right } => {
                let left_ty = self.visit_node(left)?;
                let right_ty = self.visit_node(right)?;
                self.check_binary_op(*op, &left_ty, &right_ty)
            }
            AstNode::UnaryOp { op, operand } => {
                let operand_ty = self.visit_node(operand)?;
                self.check_unary_op(*op, &operand_ty)
            }
            AstNode::Assign { target, value } => {
                let target_ty = self.visit_node(target)?;
                let value_ty = self.visit_node(value)?;
                if !target_ty.is_assignable_from(&value_ty) {
                    return Err(OmegaError::TypeError {
                        message: format!("Cannot assign {} to {}", value_ty, target_ty),
                        span: None,
                    });
                }
                Ok(target_ty)
            }
            AstNode::Call { function, args, .. } => {
                let func_ty = self.visit_node(function)?;
                match func_ty {
                    OmegaType::Function { params, return_type, .. } => {
                        if args.len() != params.len() {
                            return Err(OmegaError::TypeError {
                                message: format!("Expected {} arguments, got {}", params.len(), args.len()),
                                span: None,
                            });
                        }
                        for (arg, param_ty) in args.iter().zip(params.iter()) {
                            let arg_ty = self.visit_node(arg)?;
                            if !param_ty.is_assignable_from(&arg_ty) {
                                return Err(OmegaError::TypeError {
                                    message: format!("Expected {}, got {}", param_ty, arg_ty),
                                    span: None,
                                });
                            }
                        }
                        Ok(*return_type)
                    }
                    _ => Err(OmegaError::TypeError {
                        message: format!("Cannot call non-function type {}", func_ty),
                        span: None,
                    }),
                }
            }
            AstNode::If { condition, then_branch, elif_branches, else_branch } => {
                let cond_ty = self.visit_node(condition)?;
                if cond_ty != OmegaType::Bool {
                    return Err(OmegaError::TypeError {
                        message: format!("Expected bool, got {}", cond_ty),
                        span: None,
                    });
                }

                let then_ty = self.visit_node(then_branch)?;

                for (elif_cond, elif_body) in elif_branches {
                    let elif_cond_ty = self.visit_node(elif_cond)?;
                    if elif_cond_ty != OmegaType::Bool {
                        return Err(OmegaError::TypeError {
                            message: format!("Expected bool, got {}", elif_cond_ty),
                            span: None,
                        });
                    }
                    self.visit_node(elif_body)?;
                }

                if let Some(else_b) = else_branch {
                    self.visit_node(else_b)?;
                }

                Ok(then_ty)
            }
            AstNode::While { condition, body } => {
                let cond_ty = self.visit_node(condition)?;
                if cond_ty != OmegaType::Bool {
                    return Err(OmegaError::TypeError {
                        message: format!("Expected bool, got {}", cond_ty),
                        span: None,
                    });
                }

                let old_in_loop = self.in_loop;
                self.in_loop = true;
                self.visit_node(body)?;
                self.in_loop = old_in_loop;

                Ok(OmegaType::None)
            }
            AstNode::For { variable, iterable, body } => {
                let iter_ty = self.visit_node(iterable)?;
                let elem_ty = match iter_ty {
                    OmegaType::Array(inner, _) => *inner,
                    OmegaType::Iterator(inner) => *inner,
                    OmegaType::String => OmegaType::Char,
                    _ => OmegaType::Any,
                };

                self.scope.define_variable(variable.clone(), elem_ty, true);
                self.scope.initialize_variable(variable);

                let old_in_loop = self.in_loop;
                self.in_loop = true;
                self.visit_node(body)?;
                self.in_loop = old_in_loop;

                Ok(OmegaType::None)
            }
            AstNode::Loop { body } => {
                let old_in_loop = self.in_loop;
                self.in_loop = true;
                self.visit_node(body)?;
                self.in_loop = old_in_loop;
                Ok(OmegaType::Never)
            }
            AstNode::Break { value } => {
                if !self.in_loop {
                    return Err(OmegaError::TypeError {
                        message: "break outside of loop".to_string(),
                        span: None,
                    });
                }
                if let Some(v) = value {
                    self.visit_node(v)?;
                }
                Ok(OmegaType::Never)
            }
            AstNode::Continue => {
                if !self.in_loop {
                    return Err(OmegaError::TypeError {
                        message: "continue outside of loop".to_string(),
                        span: None,
                    });
                }
                Ok(OmegaType::Never)
            }
            AstNode::Return { value } => {
                if !self.in_function {
                    return Err(OmegaError::TypeError {
                        message: "return outside of function".to_string(),
                        span: None,
                    });
                }
                if let Some(v) = value {
                    let val_ty = self.visit_node(v)?;
                    if let Some(ret_ty) = &self.current_return_type {
                        if !ret_ty.is_assignable_from(&val_ty) {
                            return Err(OmegaError::TypeError {
                                message: format!("Expected return type {}, got {}", ret_ty, val_ty),
                                span: None,
                            });
                        }
                    }
                }
                Ok(OmegaType::Never)
            }
            AstNode::FunctionDef { name, params, return_type, body, is_async, .. } => {
                let ret_ty = if let Some(ann) = return_type {
                    self.resolve_type(ann)?
                } else {
                    OmegaType::None
                };

                let param_infos: Vec<(String, OmegaType)> = params.iter().map(|p| {
                    let ty = p.type_annotation.as_ref()
                        .map(|ann| self.resolve_type(ann).unwrap_or(OmegaType::Any))
                        .unwrap_or(OmegaType::Any);
                    (p.name.clone(), ty)
                }).collect();

                self.scope.define_function(name.clone(), FunctionInfo {
                    name: name.clone(),
                    params: param_infos.clone(),
                    return_type: ret_ty.clone(),
                    is_async: *is_async,
                    is_pub: false,
                });

                let old_return = self.current_return_type.clone();
                let old_in_function = self.in_function;
                self.current_return_type = Some(ret_ty.clone());
                self.in_function = true;

                self.scope = self.scope.child(ScopeType::Function);
                for (param_name, param_ty) in &param_infos {
                    self.scope.define_variable(param_name.clone(), param_ty.clone(), true);
                    self.scope.initialize_variable(param_name);
                }
                self.visit_node(body)?;
                self.check_unused();
                self.scope = *self.scope.parent.take().unwrap_or(Box::new(Scope::global()));

                self.current_return_type = old_return;
                self.in_function = old_in_function;

                Ok(OmegaType::Function {
                    params: param_infos.into_iter().map(|(_, t)| t).collect(),
                    return_type: Box::new(ret_ty),
                    is_async: *is_async,
                })
            }
            AstNode::StructDef { name, fields, .. } => {
                let field_types: Vec<(String, OmegaType)> = fields.iter().map(|f| {
                    let ty = self.resolve_type(&f.type_annotation).unwrap_or(OmegaType::Any);
                    (f.name.clone(), ty)
                }).collect();

                self.type_registry.register_struct(name, field_types);
                self.scope.define_type(name.clone(), OmegaType::Struct(name.clone(), Vec::new()));
                Ok(OmegaType::None)
            }
            AstNode::Array(elements) => {
                if elements.is_empty() {
                    return Ok(OmegaType::Array(Box::new(OmegaType::Any), Some(0)));
                }
                let first_ty = self.visit_node(&elements[0])?;
                for elem in &elements[1..] {
                    let elem_ty = self.visit_node(elem)?;
                    if first_ty.unify(&elem_ty).is_none() {
                        return Err(OmegaError::TypeError {
                            message: format!("Array elements have incompatible types: {} and {}", first_ty, elem_ty),
                            span: None,
                        });
                    }
                }
                Ok(OmegaType::Array(Box::new(first_ty), Some(elements.len())))
            }
            AstNode::Map(entries) => {
                if entries.is_empty() {
                    return Ok(OmegaType::Map(Box::new(OmegaType::Any), Box::new(OmegaType::Any)));
                }
                let key_ty = self.visit_node(&entries[0].0)?;
                let val_ty = self.visit_node(&entries[0].1)?;
                for (k, v) in &entries[1..] {
                    let kt = self.visit_node(k)?;
                    let vt = self.visit_node(v)?;
                    if key_ty.unify(&kt).is_none() {
                        return Err(OmegaError::TypeError {
                            message: format!("Map keys have incompatible types: {} and {}", key_ty, kt),
                            span: None,
                        });
                    }
                    if val_ty.unify(&vt).is_none() {
                        return Err(OmegaError::TypeError {
                            message: format!("Map values have incompatible types: {} and {}", val_ty, vt),
                            span: None,
                        });
                    }
                }
                Ok(OmegaType::Map(Box::new(key_ty), Box::new(val_ty)))
            }
            AstNode::Tuple(elements) => {
                let types: Vec<OmegaType> = elements.iter()
                    .map(|e| self.visit_node(e))
                    .collect::<OmegaResult<Vec<_>>>()?;
                Ok(OmegaType::Tuple(types))
            }
            AstNode::Print { args, .. } => {
                for arg in args {
                    self.visit_node(arg)?;
                }
                Ok(OmegaType::None)
            }
            AstNode::Assert { condition, message } => {
                let cond_ty = self.visit_node(condition)?;
                if cond_ty != OmegaType::Bool {
                    return Err(OmegaError::TypeError {
                        message: format!("Expected bool, got {}", cond_ty),
                        span: None,
                    });
                }
                if let Some(msg) = message {
                    self.visit_node(msg)?;
                }
                Ok(OmegaType::None)
            }
            _ => Ok(OmegaType::Any),
        }
    }

    fn check_binary_op(&self, op: BinaryOp, left: &OmegaType, right: &OmegaType) -> OmegaResult<OmegaType> {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod | BinaryOp::Pow | BinaryOp::FloorDiv => {
                if left.is_numeric() && right.is_numeric() {
                    Ok(left.unify(right).unwrap_or(OmegaType::Float64))
                } else if *left == OmegaType::String && *right == OmegaType::String && op == BinaryOp::Add {
                    Ok(OmegaType::String)
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Cannot apply {} to {} and {}", op, left, right),
                        span: None,
                    })
                }
            }
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
                if left.is_integer() && right.is_integer() {
                    Ok(left.clone())
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Bitwise operator {} requires integer types", op),
                        span: None,
                    })
                }
            }
            BinaryOp::Eq | BinaryOp::Ne => Ok(OmegaType::Bool),
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Spaceship => {
                if left.is_numeric() && right.is_numeric() {
                    Ok(OmegaType::Bool)
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Cannot compare {} and {}", left, right),
                        span: None,
                    })
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                if *left == OmegaType::Bool && *right == OmegaType::Bool {
                    Ok(OmegaType::Bool)
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Logical operator {} requires bool types", op),
                        span: None,
                    })
                }
            }
            _ => Ok(OmegaType::Any),
        }
    }

    fn check_unary_op(&self, op: UnaryOp, operand: &OmegaType) -> OmegaResult<OmegaType> {
        match op {
            UnaryOp::Neg => {
                if operand.is_numeric() {
                    Ok(operand.clone())
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Cannot negate {}", operand),
                        span: None,
                    })
                }
            }
            UnaryOp::Not => {
                if *operand == OmegaType::Bool {
                    Ok(OmegaType::Bool)
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Cannot apply 'not' to {}", operand),
                        span: None,
                    })
                }
            }
            UnaryOp::BitNot => {
                if operand.is_integer() {
                    Ok(operand.clone())
                } else {
                    Err(OmegaError::TypeError {
                        message: format!("Cannot apply bitwise not to {}", operand),
                        span: None,
                    })
                }
            }
            _ => Ok(operand.clone()),
        }
    }

    fn resolve_type(&self, ann: &TypeAnnotation) -> OmegaResult<OmegaType> {
        match &ann.kind {
            TypeAnnotationKind::Simple(name) => {
                self.type_registry.get_type(name)
                    .cloned()
                    .or_else(|| self.scope.get_type(name).cloned())
                    .ok_or_else(|| OmegaError::TypeError {
                        message: format!("Unknown type '{}'", name),
                        span: None,
                    })
            }
            TypeAnnotationKind::Generic { base, args } => {
                let base_ty = self.resolve_type(base)?;
                let arg_tys: Vec<OmegaType> = args.iter().map(|a| self.resolve_type(a)).collect::<OmegaResult<Vec<_>>>()?;
                match base_ty {
                    OmegaType::Map(_, _) if arg_tys.len() == 2 => {
                        Ok(OmegaType::Map(Box::new(arg_tys[0].clone()), Box::new(arg_tys[1].clone())))
                    }
                    OmegaType::Array(inner, _) => Ok(OmegaType::Array(inner, None)),
                    _ => Ok(OmegaType::Generic(base_ty.to_string(), arg_tys)),
                }
            }
            TypeAnnotationKind::Tuple(types) => {
                let resolved: Vec<OmegaType> = types.iter().map(|t| self.resolve_type(t)).collect::<OmegaResult<Vec<_>>>()?;
                Ok(OmegaType::Tuple(resolved))
            }
            TypeAnnotationKind::Array { element, .. } => {
                let elem_ty = self.resolve_type(element)?;
                Ok(OmegaType::Array(Box::new(elem_ty), None))
            }
            TypeAnnotationKind::Optional(inner) => {
                let inner_ty = self.resolve_type(inner)?;
                Ok(OmegaType::Optional(Box::new(inner_ty)))
            }
            TypeAnnotationKind::Reference { mutable, inner } => {
                let inner_ty = self.resolve_type(inner)?;
                Ok(OmegaType::Reference { mutable: *mutable, inner: Box::new(inner_ty) })
            }
            TypeAnnotationKind::Function { params, return_type } => {
                let param_tys: Vec<OmegaType> = params.iter().map(|p| self.resolve_type(p)).collect::<OmegaResult<Vec<_>>>()?;
                let ret_ty = self.resolve_type(return_type)?;
                Ok(OmegaType::Function {
                    params: param_tys,
                    return_type: Box::new(ret_ty),
                    is_async: false,
                })
            }
            TypeAnnotationKind::SelfType => Ok(OmegaType::SelfType),
            TypeAnnotationKind::Infer => Ok(OmegaType::Any),
            TypeAnnotationKind::Never => Ok(OmegaType::Never),
            _ => Ok(OmegaType::Any),
        }
    }

    fn check_unused(&self) {
        for var in self.scope.unused_variables() {
            self.diagnostics.report(Diagnostic::warning(
                format!("Unused variable '{}'", var.name)
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn analyze(source: &str) -> OmegaResult<()> {
        let mut parser = Parser::new(source);
        let ast = parser.parse()?;
        let mut analyzer = SemanticAnalyzer::new(source);
        analyzer.analyze(&ast)
    }

    #[test]
    fn test_let_binding() {
        assert!(analyze("let x = 42").is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        let result = analyze("let x: bool = 42");
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_op() {
        assert!(analyze("let x = 1 + 2").is_ok());
    }

    #[test]
    fn test_function() {
        assert!(analyze("fn add(a: i32, b: i32) -> i32 { return a + b }").is_ok());
    }
}
