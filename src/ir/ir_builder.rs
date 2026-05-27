use super::ir_node::*;
use crate::ast::*;
use crate::errors::OmegaResult;

pub struct IrBuilder {
    nodes: Vec<IrNode>,
    temp_counter: usize,
    label_counter: usize,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            temp_counter: 0,
            label_counter: 0,
        }
    }

    pub fn build_from_ast(&mut self, ast: &AstNode) -> OmegaResult<Vec<IrNode>> {
        self.nodes.clear();
        self.visit_node(ast)?;
        Ok(std::mem::take(&mut self.nodes))
    }

    fn new_temp(&mut self) -> String {
        self.temp_counter += 1;
        format!("_t{}", self.temp_counter)
    }

    fn new_label(&mut self) -> String {
        self.label_counter += 1;
        format!("_L{}", self.label_counter)
    }

    fn visit_node(&mut self, node: &AstNode) -> OmegaResult<IrNode> {
        let ir = match node {
            AstNode::Program(stmts) | AstNode::Block(stmts) => {
                let mut body = Vec::new();
                for stmt in stmts {
                    body.push(self.visit_node(stmt)?);
                }
                IrNode::Block(body)
            }

            AstNode::IntegerLiteral(n) => IrNode::ConstInteger(*n),
            AstNode::FloatLiteral(n) => IrNode::ConstFloat(*n),
            AstNode::BoolLiteral(b) => IrNode::ConstBool(*b),
            AstNode::StringLiteral(s) => IrNode::ConstString(s.clone()),
            AstNode::NoneLiteral => IrNode::ConstNone,
            AstNode::Identifier(name) => IrNode::LoadLocal(name.clone()),

            AstNode::LetBinding {
                name,
                mutable: _,
                type_annotation: _,
                value,
            } => {
                let init = if let Some(val) = value {
                    self.visit_node(val)?
                } else {
                    IrNode::ConstNone
                };
                IrNode::StoreLocal(name.clone(), Box::new(init))
            }

            AstNode::ConstBinding {
                name,
                type_annotation: _,
                value,
            } => {
                let init = self.visit_node(value)?;
                IrNode::StoreLocal(name.clone(), Box::new(init))
            }

            AstNode::Assign { target, value } => {
                let val = self.visit_node(value)?;
                match target.as_ref() {
                    AstNode::Identifier(name) => IrNode::StoreLocal(name.clone(), Box::new(val)),
                    _ => {
                        let tgt = self.visit_node(target)?;
                        IrNode::StoreLocal("_assign_target".to_string(), Box::new(val))
                    }
                }
            }

            AstNode::BinaryOp { op, left, right } => {
                let l = self.visit_node(left)?;
                let r = self.visit_node(right)?;
                match op {
                    BinaryOp::Add => IrNode::Add(Box::new(l), Box::new(r)),
                    BinaryOp::Sub => IrNode::Sub(Box::new(l), Box::new(r)),
                    BinaryOp::Mul => IrNode::Mul(Box::new(l), Box::new(r)),
                    BinaryOp::Div => IrNode::Div(Box::new(l), Box::new(r)),
                    BinaryOp::Mod => IrNode::Mod(Box::new(l), Box::new(r)),
                    BinaryOp::Pow => IrNode::Pow(Box::new(l), Box::new(r)),
                    BinaryOp::Eq => IrNode::Eq(Box::new(l), Box::new(r)),
                    BinaryOp::Ne => IrNode::Ne(Box::new(l), Box::new(r)),
                    BinaryOp::Lt => IrNode::Lt(Box::new(l), Box::new(r)),
                    BinaryOp::Le => IrNode::Le(Box::new(l), Box::new(r)),
                    BinaryOp::Gt => IrNode::Gt(Box::new(l), Box::new(r)),
                    BinaryOp::Ge => IrNode::Ge(Box::new(l), Box::new(r)),
                    BinaryOp::And => IrNode::And(Box::new(l), Box::new(r)),
                    BinaryOp::Or => IrNode::Or(Box::new(l), Box::new(r)),
                    BinaryOp::BitAnd => IrNode::BitAnd(Box::new(l), Box::new(r)),
                    BinaryOp::BitOr => IrNode::BitOr(Box::new(l), Box::new(r)),
                    BinaryOp::BitXor => IrNode::BitXor(Box::new(l), Box::new(r)),
                    BinaryOp::Shl => IrNode::Shl(Box::new(l), Box::new(r)),
                    BinaryOp::Shr => IrNode::Shr(Box::new(l), Box::new(r)),
                    _ => IrNode::Nop,
                }
            }

            AstNode::UnaryOp { op, operand } => {
                let val = self.visit_node(operand)?;
                match op {
                    UnaryOp::Neg => IrNode::Neg(Box::new(val)),
                    UnaryOp::Not => IrNode::Not(Box::new(val)),
                    UnaryOp::BitNot => IrNode::BitNot(Box::new(val)),
                    _ => IrNode::Nop,
                }
            }

            AstNode::Call { function, args, .. } => {
                let func = self.visit_node(function)?;
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.visit_node(arg)?);
                }
                IrNode::Call {
                    function: Box::new(func),
                    args: ir_args,
                }
            }

            AstNode::If {
                condition,
                then_branch,
                elif_branches,
                else_branch,
            } => {
                let cond = self.visit_node(condition)?;
                let then_body = vec![self.visit_node(then_branch)?];
                let else_body = if let Some(else_b) = else_branch {
                    vec![self.visit_node(else_b)?]
                } else {
                    Vec::new()
                };
                IrNode::If {
                    condition: Box::new(cond),
                    then_branch: then_body,
                    else_branch: else_body,
                }
            }

            AstNode::While { condition, body } => {
                let cond = self.visit_node(condition)?;
                let body_ir = vec![self.visit_node(body)?];
                IrNode::While {
                    condition: Box::new(cond),
                    body: body_ir,
                }
            }

            AstNode::For {
                variable,
                iterable,
                body,
            } => {
                let iter = self.visit_node(iterable)?;
                let body_ir = vec![self.visit_node(body)?];
                IrNode::For {
                    variable: variable.clone(),
                    iterable: Box::new(iter),
                    body: body_ir,
                }
            }

            AstNode::Return { value } => {
                let val = value.as_ref().map(|v| Box::new(self.visit_node(v).unwrap()));
                IrNode::Return(val)
            }

            AstNode::Break => IrNode::Break,
            AstNode::Continue => IrNode::Continue,

            AstNode::Array(elements) => {
                let mut ir_elements = Vec::new();
                for elem in elements {
                    ir_elements.push(self.visit_node(elem)?);
                }
                IrNode::Array(ir_elements)
            }

            AstNode::Map(entries) => {
                let mut ir_entries = Vec::new();
                for (k, v) in entries {
                    ir_entries.push((self.visit_node(k)?, self.visit_node(v)?));
                }
                IrNode::Map(ir_entries)
            }

            AstNode::Tuple(elements) => {
                let mut ir_elements = Vec::new();
                for elem in elements {
                    ir_elements.push(self.visit_node(elem)?);
                }
                IrNode::Tuple(ir_elements)
            }

            AstNode::Print { args, newline } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.visit_node(arg)?);
                }
                IrNode::Print(ir_args, *newline)
            }

            _ => IrNode::Nop,
        };

        Ok(ir)
    }

    pub fn optimize(&mut self, nodes: &mut Vec<IrNode>) {
        // Constant folding
        for node in nodes.iter_mut() {
            self.fold_constants(node);
        }
    }

    fn fold_constants(&self, node: &mut IrNode) {
        match node {
            IrNode::Add(l, r) => {
                if let (IrNode::ConstInteger(a), IrNode::ConstInteger(b)) = (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstInteger(a + b);
                } else if let (IrNode::ConstFloat(a), IrNode::ConstFloat(b)) =
                    (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstFloat(a + b);
                }
            }
            IrNode::Sub(l, r) => {
                if let (IrNode::ConstInteger(a), IrNode::ConstInteger(b)) = (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstInteger(a - b);
                } else if let (IrNode::ConstFloat(a), IrNode::ConstFloat(b)) =
                    (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstFloat(a - b);
                }
            }
            IrNode::Mul(l, r) => {
                if let (IrNode::ConstInteger(a), IrNode::ConstInteger(b)) = (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstInteger(a * b);
                } else if let (IrNode::ConstFloat(a), IrNode::ConstFloat(b)) =
                    (l.as_ref(), r.as_ref())
                {
                    *node = IrNode::ConstFloat(a * b);
                }
            }
            IrNode::Div(l, r) => {
                if let (IrNode::ConstInteger(a), IrNode::ConstInteger(b)) = (l.as_ref(), r.as_ref())
                {
                    if *b != 0 {
                        *node = IrNode::ConstInteger(a / b);
                    }
                } else if let (IrNode::ConstFloat(a), IrNode::ConstFloat(b)) =
                    (l.as_ref(), r.as_ref())
                {
                    if *b != 0.0 {
                        *node = IrNode::ConstFloat(a / b);
                    }
                }
            }
            IrNode::Neg(n) => {
                if let IrNode::ConstInteger(v) = n.as_ref() {
                    *node = IrNode::ConstInteger(-v);
                } else if let IrNode::ConstFloat(v) = n.as_ref() {
                    *node = IrNode::ConstFloat(-v);
                }
            }
            IrNode::Not(n) => {
                if let IrNode::ConstBool(v) = n.as_ref() {
                    *node = IrNode::ConstBool(!v);
                }
            }
            _ => {}
        }
    }
}
