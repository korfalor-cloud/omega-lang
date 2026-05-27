use crate::ast::*;
use crate::errors::{Diagnostic, DiagnosticBag, OmegaError, OmegaResult, Position, Span};
use crate::lexer::{Scanner, Token, TokenKind, Keyword};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    diagnostics: DiagnosticBag,
    source: String,
}

impl Parser {
    pub fn new(source: &str) -> Self {
        let mut scanner = Scanner::new(source);
        let tokens = scanner.scan_all();
        Self {
            tokens,
            current: 0,
            diagnostics: DiagnosticBag::new(source, "<input>"),
            source: source.to_string(),
        }
    }

    pub fn parse(&mut self) -> OmegaResult<AstNode> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            self.skip_newlines();
            if self.is_at_end() {
                break;
            }
            match self.parse_top_level() {
                Ok(item) => items.push(item),
                Err(e) => {
                    self.diagnostics.report(Diagnostic::error(e.to_string()));
                    self.synchronize();
                }
            }
        }

        Ok(AstNode::Program(items))
    }

    fn parse_top_level(&mut self) -> OmegaResult<AstNode> {
        // Check for decorators
        while self.check(&TokenKind::At) {
            self.advance(); // consume @
            self.consume_identifier("Expected decorator name")?;
            if self.check(&TokenKind::LeftParen) {
                self.parse_call_args()?;
            }
            self.skip_newlines();
        }

        // Check for visibility modifier
        let is_pub = if self.check_keyword(Keyword::Pub) {
            self.advance();
            self.skip_newlines();
            true
        } else {
            false
        };

        // Check for async
        let is_async = if self.check_keyword(Keyword::Async) {
            self.advance();
            self.skip_newlines();
            true
        } else {
            false
        };

        match &self.peek().kind {
            TokenKind::Keyword(Keyword::Fn) => self.parse_function_def(is_pub, is_async),
            TokenKind::Keyword(Keyword::Struct) => self.parse_struct_def(is_pub),
            TokenKind::Keyword(Keyword::Enum) => self.parse_enum_def(is_pub),
            TokenKind::Keyword(Keyword::Trait) => self.parse_trait_def(is_pub),
            TokenKind::Keyword(Keyword::Impl) => self.parse_impl_block(),
            TokenKind::Keyword(Keyword::Type) => self.parse_type_alias(is_pub),
            TokenKind::Keyword(Keyword::Mod) => self.parse_module(),
            TokenKind::Keyword(Keyword::Use) => self.parse_use_decl(),
            TokenKind::Keyword(Keyword::Test) => self.parse_test_block(),
            _ => self.parse_statement(),
        }
    }

    fn parse_function_def(&mut self, is_pub: bool, is_async: bool) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Fn, "Expected 'fn'")?;
        let name = self.consume_identifier("Expected function name")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        self.consume(&TokenKind::LeftParen, "Expected '(' after function name")?;
        let params = self.parse_params()?;
        self.consume(&TokenKind::RightParen, "Expected ')' after parameters")?;

        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        let body = if self.check(&TokenKind::LeftBrace) {
            self.parse_block()?
        } else if self.check(&TokenKind::FatArrow) {
            self.advance();
            self.parse_expression()?
        } else {
            return Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: "Expected '{' or '=>' after function signature".to_string(),
            });
        };

        Ok(AstNode::FunctionDef {
            name,
            type_params,
            params,
            return_type,
            body: Box::new(body),
            is_async,
            is_pub,
        })
    }

    fn parse_struct_def(&mut self, is_pub: bool) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Struct, "Expected 'struct'")?;
        let name = self.consume_identifier("Expected struct name")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;
        let mut fields = Vec::new();
        let mut methods = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();

            if self.check_keyword(Keyword::Fn) || self.check_keyword(Keyword::Pub) {
                methods.push(self.parse_top_level()?);
            } else {
                let field_pub = if self.check_keyword(Keyword::Pub) {
                    self.advance();
                    true
                } else {
                    false
                };

                let field_mut = if self.check_keyword(Keyword::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };

                let field_name = self.consume_identifier("Expected field name")?;
                self.consume(&TokenKind::Colon, "Expected ':' after field name")?;
                let type_annotation = self.parse_type_annotation()?;

                let default = if self.check(&TokenKind::Equal) {
                    self.advance();
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };

                fields.push(StructField {
                    name: field_name,
                    type_annotation,
                    default,
                    is_pub: field_pub,
                    is_mut: field_mut,
                });

                if self.check(&TokenKind::Comma) {
                    self.advance();
                }
            }
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::StructDef {
            name,
            type_params,
            fields,
            methods,
            is_pub,
        })
    }

    fn parse_enum_def(&mut self, is_pub: bool) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Enum, "Expected 'enum'")?;
        let name = self.consume_identifier("Expected enum name")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;
        let mut variants = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            let variant_name = self.consume_identifier("Expected variant name")?;

            let data = if self.check(&TokenKind::LeftParen) {
                self.advance();
                let mut types = Vec::new();
                while !self.check(&TokenKind::RightParen) {
                    types.push(self.parse_type_annotation()?);
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.consume(&TokenKind::RightParen, "Expected ')'")?;
                Some(EnumVariantData::Tuple(types))
            } else if self.check(&TokenKind::LeftBrace) {
                self.advance();
                let mut fields = Vec::new();
                while !self.check(&TokenKind::RightBrace) {
                    let fname = self.consume_identifier("Expected field name")?;
                    self.consume(&TokenKind::Colon, "Expected ':'")?;
                    let ftype = self.parse_type_annotation()?;
                    fields.push(StructField {
                        name: fname,
                        type_annotation: ftype,
                        default: None,
                        is_pub: false,
                        is_mut: false,
                    });
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.consume(&TokenKind::RightBrace, "Expected '}'")?;
                Some(EnumVariantData::Struct(fields))
            } else if self.check(&TokenKind::Equal) {
                self.advance();
                Some(EnumVariantData::Value(Box::new(self.parse_expression()?)))
            } else {
                None
            };

            variants.push(EnumVariant { name: variant_name, data });

            if self.check(&TokenKind::Comma) {
                self.advance();
            }
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::EnumDef {
            name,
            type_params,
            variants,
            is_pub,
        })
    }

    fn parse_trait_def(&mut self, is_pub: bool) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Trait, "Expected 'trait'")?;
        let name = self.consume_identifier("Expected trait name")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        let supertraits = if self.check(&TokenKind::Colon) {
            self.advance();
            let mut traits = Vec::new();
            loop {
                traits.push(self.consume_identifier("Expected trait name")?);
                if !self.check(&TokenKind::Plus) {
                    break;
                }
                self.advance();
            }
            traits
        } else {
            Vec::new()
        };

        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;
        let mut items = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();

            if self.check_keyword(Keyword::Fn) {
                self.advance();
                let method_name = self.consume_identifier("Expected method name")?;
                self.consume(&TokenKind::LeftParen, "Expected '('")?;
                let params = self.parse_params()?;
                self.consume(&TokenKind::RightParen, "Expected ')'")?;

                let return_type = if self.check(&TokenKind::Arrow) {
                    self.advance();
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };

                let body = if self.check(&TokenKind::LeftBrace) {
                    Some(Box::new(self.parse_block()?))
                } else {
                    None
                };

                items.push(TraitItem::Method {
                    name: method_name,
                    params,
                    return_type,
                    body,
                });
            } else if self.check_keyword(Keyword::Type) {
                self.advance();
                let type_name = self.consume_identifier("Expected type name")?;
                items.push(TraitItem::AssociatedType {
                    name: type_name,
                    bounds: Vec::new(),
                    default: None,
                });
            } else if self.check_keyword(Keyword::Const) {
                self.advance();
                let const_name = self.consume_identifier("Expected const name")?;
                self.consume(&TokenKind::Colon, "Expected ':'")?;
                let type_ann = self.parse_type_annotation()?;
                let value = if self.check(&TokenKind::Equal) {
                    self.advance();
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                items.push(TraitItem::Const {
                    name: const_name,
                    type_annotation: type_ann,
                    value,
                });
            }

            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::TraitDef {
            name,
            type_params,
            supertraits,
            items,
            is_pub,
        })
    }

    fn parse_impl_block(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Impl, "Expected 'impl'")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        let mut trait_name = None;
        let target_type = self.parse_type_annotation()?;

        if self.check_keyword(Keyword::For) {
            self.advance();
            trait_name = Some(match &target_type.kind {
                TypeAnnotationKind::Simple(name) => name.clone(),
                _ => return Err(OmegaError::ParseError {
                    location: format!("{}:{}", self.peek().line, self.peek().col),
                    message: "Expected trait name".to_string(),
                }),
            });
        }

        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;
        let mut items = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            if !self.check(&TokenKind::RightBrace) {
                items.push(self.parse_top_level()?);
            }
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::ImplBlock {
            type_params,
            trait_name,
            target_type,
            items,
        })
    }

    fn parse_type_alias(&mut self, is_pub: bool) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Type, "Expected 'type'")?;
        let name = self.consume_identifier("Expected type name")?;

        let type_params = if self.check(&TokenKind::Less) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };

        self.consume(&TokenKind::Equal, "Expected '=' in type alias")?;
        let value = self.parse_type_annotation()?;

        Ok(AstNode::TypeAlias {
            name,
            type_params,
            value,
            is_pub,
        })
    }

    fn parse_module(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Mod, "Expected 'mod'")?;
        let name = self.consume_identifier("Expected module name")?;

        if self.check(&TokenKind::Semicolon) {
            self.advance();
            return Ok(AstNode::Module { name, body: Vec::new() });
        }

        self.consume(&TokenKind::LeftBrace, "Expected '{' or ';'")?;
        let mut body = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            if !self.check(&TokenKind::RightBrace) {
                body.push(self.parse_top_level()?);
            }
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::Module { name, body })
    }

    fn parse_use_decl(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Use, "Expected 'use'")?;

        let mut path = Vec::new();
        path.push(self.consume_identifier("Expected module path")?);

        while self.check(&TokenKind::Dot) {
            self.advance();
            path.push(self.consume_identifier("Expected identifier after '.'")?);
        }

        let (alias, items) = if self.check(&TokenKind::As) {
            self.advance();
            (Some(self.consume_identifier("Expected alias")?), None)
        } else if self.check(&TokenKind::LeftBrace) {
            self.advance();
            let mut import_items = Vec::new();
            while !self.check(&TokenKind::RightBrace) {
                let item_name = self.consume_identifier("Expected item name")?;
                let item_alias = if self.check(&TokenKind::As) {
                    self.advance();
                    Some(self.consume_identifier("Expected alias")?)
                } else {
                    None
                };
                import_items.push((item_name, item_alias));
                if self.check(&TokenKind::Comma) {
                    self.advance();
                }
            }
            self.consume(&TokenKind::RightBrace, "Expected '}'")?;
            (None, Some(import_items))
        } else {
            (None, None)
        };

        if self.check(&TokenKind::Semicolon) {
            self.advance();
        }

        Ok(AstNode::UseDecl { path, alias, items })
    }

    fn parse_test_block(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Test, "Expected 'test'")?;
        let name = self.consume_identifier("Expected test name")?;
        let body = self.parse_block()?;

        Ok(AstNode::TestBlock {
            name,
            body: Box::new(body),
        })
    }

    fn parse_statement(&mut self) -> OmegaResult<AstNode> {
        match &self.peek().kind {
            TokenKind::Keyword(Keyword::Let) => self.parse_let_binding(),
            TokenKind::Keyword(Keyword::Const) => self.parse_const_binding(),
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::While) => self.parse_while(),
            TokenKind::Keyword(Keyword::For) => self.parse_for(),
            TokenKind::Keyword(Keyword::Loop) => self.parse_loop(),
            TokenKind::Keyword(Keyword::Break) => self.parse_break(),
            TokenKind::Keyword(Keyword::Continue) => {
                self.advance();
                Ok(AstNode::Continue)
            }
            TokenKind::Keyword(Keyword::Return) => self.parse_return(),
            TokenKind::Keyword(Keyword::Throw) => self.parse_throw(),
            TokenKind::Keyword(Keyword::Try) => self.parse_try_catch(),
            TokenKind::Keyword(Keyword::Defer) => self.parse_defer(),
            TokenKind::Keyword(Keyword::Errdefer) => self.parse_errdefer(),
            TokenKind::Keyword(Keyword::Assert) => self.parse_assert(),
            TokenKind::Keyword(Keyword::Print) => self.parse_print(false),
            TokenKind::Keyword(Keyword::Println) => self.parse_print(true),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_binding(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Let, "Expected 'let'")?;

        let mutable = if self.check_keyword(Keyword::Mut) {
            self.advance();
            true
        } else {
            false
        };

        let name = self.consume_identifier("Expected variable name")?;

        let type_annotation = if self.check(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        let value = if self.check(&TokenKind::Equal) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        Ok(AstNode::LetBinding {
            name,
            mutable,
            type_annotation,
            value,
        })
    }

    fn parse_const_binding(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Const, "Expected 'const'")?;
        let name = self.consume_identifier("Expected constant name")?;

        let type_annotation = if self.check(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        self.consume(&TokenKind::Equal, "Expected '=' in const declaration")?;
        let value = self.parse_expression()?;

        Ok(AstNode::ConstBinding {
            name,
            type_annotation,
            value: Box::new(value),
        })
    }

    fn parse_if(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::If, "Expected 'if'")?;
        let condition = self.parse_expression()?;
        let then_branch = self.parse_block()?;

        let mut elif_branches = Vec::new();
        while self.check_keyword(Keyword::Else) {
            self.advance();
            if self.check_keyword(Keyword::If) {
                self.advance();
                let elif_cond = self.parse_expression()?;
                let elif_body = self.parse_block()?;
                elif_branches.push((elif_cond, elif_body));
            } else {
                let else_body = self.parse_block()?;
                return Ok(AstNode::If {
                    condition,
                    then_branch: Box::new(then_branch),
                    elif_branches,
                    else_branch: Some(Box::new(else_body)),
                });
            }
        }

        Ok(AstNode::If {
            condition,
            then_branch: Box::new(then_branch),
            elif_branches,
            else_branch: None,
        })
    }

    fn parse_while(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::While, "Expected 'while'")?;
        let condition = self.parse_expression()?;
        let body = self.parse_block()?;

        Ok(AstNode::While {
            condition,
            body: Box::new(body),
        })
    }

    fn parse_for(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::For, "Expected 'for'")?;
        let variable = self.consume_identifier("Expected variable name")?;
        self.consume_keyword(Keyword::In, "Expected 'in'")?;
        let iterable = self.parse_expression()?;
        let body = self.parse_block()?;

        Ok(AstNode::For {
            variable,
            iterable,
            body: Box::new(body),
        })
    }

    fn parse_loop(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Loop, "Expected 'loop'")?;
        let body = self.parse_block()?;

        Ok(AstNode::Loop {
            body: Box::new(body),
        })
    }

    fn parse_break(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Break, "Expected 'break'")?;

        let value = if !self.check(&TokenKind::Newline) && !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        Ok(AstNode::Break { value })
    }

    fn parse_return(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Return, "Expected 'return'")?;

        let value = if !self.check(&TokenKind::Newline) && !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        Ok(AstNode::Return { value })
    }

    fn parse_throw(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Throw, "Expected 'throw'")?;
        let value = self.parse_expression()?;
        Ok(AstNode::Throw { value: Box::new(value) })
    }

    fn parse_try_catch(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Try, "Expected 'try'")?;
        let try_body = self.parse_block()?;

        let mut catch_clauses = Vec::new();
        while self.check_keyword(Keyword::Catch) {
            self.advance();

            let binding = if self.check(&TokenKind::LeftParen) {
                self.advance();
                let name = self.consume_identifier("Expected catch variable")?;
                self.consume(&TokenKind::RightParen, "Expected ')'")?;
                Some(name)
            } else {
                None
            };

            let type_annotation = if self.check(&TokenKind::Colon) {
                self.advance();
                Some(self.parse_type_annotation()?)
            } else {
                None
            };

            let body = self.parse_block()?;
            catch_clauses.push(CatchClause {
                binding,
                type_annotation,
                body: Box::new(body),
            });
        }

        let finally_body = if self.check_keyword(Keyword::Finally) {
            self.advance();
            Some(Box::new(self.parse_block()?))
        } else {
            None
        };

        Ok(AstNode::TryCatch {
            try_body: Box::new(try_body),
            catch_clauses,
            finally_body,
        })
    }

    fn parse_defer(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Defer, "Expected 'defer'")?;
        let body = self.parse_block()?;
        Ok(AstNode::Defer { body: Box::new(body) })
    }

    fn parse_errdefer(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Errdefer, "Expected 'errdefer'")?;
        let body = self.parse_block()?;
        Ok(AstNode::Errdefer { body: Box::new(body) })
    }

    fn parse_assert(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Assert, "Expected 'assert'")?;
        let condition = self.parse_expression()?;
        let message = if self.check(&TokenKind::Comma) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        Ok(AstNode::Assert { condition: Box::new(condition), message })
    }

    fn parse_print(&mut self, newline: bool) -> OmegaResult<AstNode> {
        if newline {
            self.consume_keyword(Keyword::Println, "Expected 'println'")?;
        } else {
            self.consume_keyword(Keyword::Print, "Expected 'print'")?;
        }
        let args = self.parse_call_args()?;
        Ok(AstNode::Print { args, newline })
    }

    fn parse_expression_statement(&mut self) -> OmegaResult<AstNode> {
        let expr = self.parse_expression()?;

        if self.check(&TokenKind::Equal) && !matches!(expr, AstNode::BinaryOp { .. }) {
            self.advance();
            let value = self.parse_expression()?;
            return Ok(AstNode::Assign {
                target: Box::new(expr),
                value: Box::new(value),
            });
        }

        if self.check_compound_assignment() {
            let op = self.get_compound_op();
            self.advance();
            let value = self.parse_expression()?;
            return Ok(AstNode::CompoundAssign {
                op,
                target: Box::new(expr),
                value: Box::new(value),
            });
        }

        Ok(expr)
    }

    fn parse_expression(&mut self) -> OmegaResult<AstNode> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_and()?;

        while self.check(&TokenKind::PipePipe) || self.check_keyword(Keyword::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = AstNode::BinaryOp {
                op: BinaryOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_not()?;

        while self.check(&TokenKind::AmpAmp) || self.check_keyword(Keyword::And) {
            self.advance();
            let right = self.parse_not()?;
            left = AstNode::BinaryOp {
                op: BinaryOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_not(&mut self) -> OmegaResult<AstNode> {
        if self.check(&TokenKind::Bang) || self.check_keyword(Keyword::Not) {
            self.advance();
            let operand = self.parse_not()?;
            return Ok(AstNode::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_bitwise_or()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::EqualEqual => BinaryOp::Eq,
                TokenKind::BangEqual => BinaryOp::Ne,
                TokenKind::Less => BinaryOp::Lt,
                TokenKind::LessEqual => BinaryOp::Le,
                TokenKind::Greater => BinaryOp::Gt,
                TokenKind::GreaterEqual => BinaryOp::Ge,
                TokenKind::Spaceship => BinaryOp::Spaceship,
                TokenKind::Keyword(Keyword::In) => BinaryOp::In,
                TokenKind::Keyword(Keyword::Is) => BinaryOp::Is,
                _ => break,
            };
            self.advance();

            if op == BinaryOp::NotIn || (op == BinaryOp::In && self.check_keyword(Keyword::Not)) {
                // handled differently
            }

            let right = self.parse_bitwise_or()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_bitwise_or(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_bitwise_xor()?;

        while self.check(&TokenKind::Pipe) {
            self.advance();
            let right = self.parse_bitwise_xor()?;
            left = AstNode::BinaryOp {
                op: BinaryOp::BitOr,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_bitwise_xor(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_bitwise_and()?;

        while self.check(&TokenKind::Caret) {
            self.advance();
            let right = self.parse_bitwise_and()?;
            left = AstNode::BinaryOp {
                op: BinaryOp::BitXor,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_bitwise_and(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_shift()?;

        while self.check(&TokenKind::Ampersand) {
            self.advance();
            let right = self.parse_shift()?;
            left = AstNode::BinaryOp {
                op: BinaryOp::BitAnd,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_shift(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_addition()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::LessLess => BinaryOp::Shl,
                TokenKind::GreaterGreater => BinaryOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_addition()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_addition(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_multiplication()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplication()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplication(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match &self.peek().kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                TokenKind::SlashSlash => BinaryOp::FloorDiv,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> OmegaResult<AstNode> {
        match &self.peek().kind {
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(AstNode::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(AstNode::UnaryOp {
                    op: UnaryOp::BitNot,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Keyword(Keyword::Async) => {
                self.advance();
                let body = self.parse_unary()?;
                Ok(AstNode::AsyncBlock { body: Box::new(body) })
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> OmegaResult<AstNode> {
        let mut left = self.parse_postfix()?;

        if self.check(&TokenKind::StarStar) {
            self.advance();
            let right = self.parse_unary()?; // right-associative
            left = AstNode::BinaryOp {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_postfix(&mut self) -> OmegaResult<AstNode> {
        let mut expr = self.parse_primary()?;

        loop {
            match &self.peek().kind {
                TokenKind::Dot => {
                    self.advance();
                    if self.check_keyword(Keyword::Await) {
                        self.advance();
                        expr = AstNode::Await { expr: Box::new(expr) };
                    } else {
                        let attr = self.consume_identifier("Expected attribute name")?;
                        if self.check(&TokenKind::LeftParen) {
                            let args = self.parse_call_args()?;
                            let kwargs = Vec::new();
                            expr = AstNode::MethodCall {
                                object: Box::new(expr),
                                method: attr,
                                args,
                                kwargs,
                            };
                        } else {
                            expr = AstNode::Attribute {
                                object: Box::new(expr),
                                attribute: attr,
                            };
                        }
                    }
                }
                TokenKind::QuestionDot => {
                    self.advance();
                    let attr = self.consume_identifier("Expected attribute name")?;
                    expr = AstNode::OptionalChain {
                        object: Box::new(expr),
                        attribute: attr,
                    };
                }
                TokenKind::LeftParen => {
                    let args = self.parse_call_args()?;
                    let kwargs = Vec::new();
                    expr = AstNode::Call {
                        function: Box::new(expr),
                        args,
                        kwargs,
                    };
                }
                TokenKind::LeftBracket => {
                    self.advance();
                    if self.check(&TokenKind::Colon) {
                        self.advance();
                        let stop = if self.check(&TokenKind::RightBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expression()?))
                        };
                        self.consume(&TokenKind::RightBracket, "Expected ']'")?;
                        expr = AstNode::Slice {
                            object: Box::new(expr),
                            start: None,
                            stop,
                            step: None,
                        };
                    } else {
                        let index = self.parse_expression()?;
                        if self.check(&TokenKind::Colon) {
                            self.advance();
                            let stop = if self.check(&TokenKind::RightBracket) {
                                None
                            } else {
                                Some(Box::new(self.parse_expression()?))
                            };
                            self.consume(&TokenKind::RightBracket, "Expected ']'")?;
                            expr = AstNode::Slice {
                                object: Box::new(expr),
                                start: Some(Box::new(index)),
                                stop,
                                step: None,
                            };
                        } else {
                            self.consume(&TokenKind::RightBracket, "Expected ']'")?;
                            expr = AstNode::Index {
                                object: Box::new(expr),
                                index: Box::new(index),
                            };
                        }
                    }
                }
                TokenKind::Keyword(Keyword::As) => {
                    self.advance();
                    let target_type = self.parse_type_annotation()?;
                    expr = AstNode::Cast {
                        expr: Box::new(expr),
                        target_type,
                    };
                }
                TokenKind::Keyword(Keyword::Is) => {
                    self.advance();
                    let check_type = self.parse_type_annotation()?;
                    expr = AstNode::TypeCheck {
                        expr: Box::new(expr),
                        check_type,
                    };
                }
                TokenKind::Question if self.check(&TokenKind::Question) => {
                    // ternary handled elsewhere
                    break;
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> OmegaResult<AstNode> {
        match &self.peek().kind.clone() {
            TokenKind::Integer(v) => {
                let v = *v;
                self.advance();
                Ok(AstNode::IntegerLiteral(v))
            }
            TokenKind::Float(v) => {
                let v = *v;
                self.advance();
                Ok(AstNode::FloatLiteral(v))
            }
            TokenKind::String(v) => {
                let v = v.clone();
                self.advance();
                Ok(AstNode::StringLiteral(v))
            }
            TokenKind::Bool(v) => {
                let v = *v;
                self.advance();
                Ok(AstNode::BoolLiteral(v))
            }
            TokenKind::Char(v) => {
                let v = *v;
                self.advance();
                Ok(AstNode::CharLiteral(v))
            }
            TokenKind::None => {
                self.advance();
                Ok(AstNode::NoneLiteral)
            }
            TokenKind::BigInt(v) => {
                let v = v.clone();
                self.advance();
                Ok(AstNode::BigIntLiteral(v))
            }
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();

                if name == "self" {
                    return Ok(AstNode::SelfExpr);
                }
                if name == "super" {
                    return Ok(AstNode::SuperExpr);
                }

                if self.check(&TokenKind::LeftBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                        let field_name = self.consume_identifier("Expected field name")?;
                        self.consume(&TokenKind::Colon, "Expected ':'")?;
                        let value = self.parse_expression()?;
                        fields.push((field_name, value));
                        if self.check(&TokenKind::Comma) {
                            self.advance();
                        }
                    }
                    self.consume(&TokenKind::RightBrace, "Expected '}'")?;
                    Ok(AstNode::StructLiteral {
                        name,
                        fields,
                        base: None,
                    })
                } else if self.check(&TokenKind::ColonColon) {
                    self.advance();
                    let variant = self.consume_identifier("Expected variant name")?;
                    let data = if self.check(&TokenKind::LeftParen) {
                        self.advance();
                        let value = self.parse_expression()?;
                        self.consume(&TokenKind::RightParen, "Expected ')'")?;
                        Some(Box::new(value))
                    } else {
                        None
                    };
                    Ok(AstNode::EnumVariant {
                        enum_name: name,
                        variant,
                        data,
                    })
                } else {
                    Ok(AstNode::Identifier(name))
                }
            }
            TokenKind::Keyword(Keyword::Fn) => self.parse_lambda(),
            TokenKind::Keyword(Keyword::If) => self.parse_if_expression(),
            TokenKind::Keyword(Keyword::Match) => self.parse_match(),
            TokenKind::Keyword(Keyword::Yield) => {
                self.advance();
                let value = if !self.check(&TokenKind::Newline) && !self.check(&TokenKind::RightBrace) {
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                Ok(AstNode::Yield { value })
            }
            TokenKind::LeftParen => self.parse_tuple_or_grouped(),
            TokenKind::LeftBracket => self.parse_array(),
            TokenKind::LeftBrace => self.parse_block(),
            TokenKind::DotDot | TokenKind::DotDotEqual => {
                // Range from start
                let inclusive = self.check(&TokenKind::DotDotEqual);
                self.advance();
                let end = self.parse_expression()?;
                let start = AstNode::IntegerLiteral(0);
                Ok(AstNode::Range {
                    start: Box::new(start),
                    end: Box::new(end),
                    inclusive,
                })
            }
            TokenKind::Keyword(Keyword::Move) => {
                self.advance();
                let operand = self.parse_primary()?;
                Ok(AstNode::UnaryOp {
                    op: UnaryOp::Move,
                    operand: Box::new(operand),
                })
            }
            TokenKind::Keyword(Keyword::Ref) => {
                self.advance();
                let mutable = if self.check_keyword(Keyword::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let operand = self.parse_primary()?;
                Ok(AstNode::UnaryOp {
                    op: UnaryOp::Ref,
                    operand: Box::new(operand),
                })
            }
            _ => Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("Unexpected token '{}'", self.peek().lexeme),
            }),
        }
    }

    fn parse_lambda(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Fn, "Expected 'fn'")?;

        let params = if self.check(&TokenKind::Pipe) {
            self.advance();
            let params = self.parse_lambda_params()?;
            self.consume(&TokenKind::Pipe, "Expected '|'")?;
            params
        } else if self.check(&TokenKind::LeftParen) {
            self.advance();
            let params = self.parse_params()?;
            self.consume(&TokenKind::RightParen, "Expected ')'")?;
            params
        } else {
            Vec::new()
        };

        let body = if self.check(&TokenKind::LeftBrace) {
            self.parse_block()?
        } else {
            self.parse_expression()?
        };

        Ok(AstNode::Lambda {
            params,
            body: Box::new(body),
            captures: Vec::new(),
        })
    }

    fn parse_lambda_params(&mut self) -> OmegaResult<Vec<Param>> {
        let mut params = Vec::new();

        while !self.check(&TokenKind::Pipe) && !self.is_at_end() {
            let name = self.consume_identifier("Expected parameter name")?;
            let type_annotation = if self.check(&TokenKind::Colon) {
                self.advance();
                Some(self.parse_type_annotation()?)
            } else {
                None
            };
            params.push(Param {
                name,
                type_annotation,
                default: None,
                is_mut: false,
                is_ref: false,
                variadic: false,
            });
            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        Ok(params)
    }

    fn parse_if_expression(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::If, "Expected 'if'")?;
        let condition = self.parse_expression()?;
        let then_branch = self.parse_block_or_expression()?;

        let mut elif_branches = Vec::new();
        while self.check_keyword(Keyword::Else) {
            self.advance();
            if self.check_keyword(Keyword::If) {
                self.advance();
                let elif_cond = self.parse_expression()?;
                let elif_body = self.parse_block_or_expression()?;
                elif_branches.push((elif_cond, elif_body));
            } else {
                let else_body = self.parse_block_or_expression()?;
                return Ok(AstNode::IfExpr {
                    condition: Box::new(condition),
                    then_branch: Box::new(then_branch),
                    elif_branches,
                    else_branch: Some(Box::new(else_body)),
                });
            }
        }

        Ok(AstNode::IfExpr {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            elif_branches,
            else_branch: None,
        })
    }

    fn parse_match(&mut self) -> OmegaResult<AstNode> {
        self.consume_keyword(Keyword::Match, "Expected 'match'")?;
        let scrutinee = self.parse_expression()?;
        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;

        let mut arms = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            let pattern = self.parse_pattern()?;
            let guard = if self.check_keyword(Keyword::If) {
                self.advance();
                Some(Box::new(self.parse_expression()?))
            } else {
                None
            };
            self.consume(&TokenKind::FatArrow, "Expected '=>'")?;
            let body = if self.check(&TokenKind::LeftBrace) {
                self.parse_block()?
            } else {
                self.parse_expression()?
            };
            arms.push(MatchArm {
                pattern,
                guard,
                body: Box::new(body),
            });
            if self.check(&TokenKind::Comma) {
                self.advance();
            }
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::MatchExpr {
            scrutinee: Box::new(scrutinee),
            arms,
        })
    }

    fn parse_pattern(&mut self) -> OmegaResult<Pattern> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                if name == "_" {
                    Ok(Pattern::Wildcard)
                } else {
                    Ok(Pattern::Identifier(name))
                }
            }
            TokenKind::Integer(v) => {
                let v = *v;
                self.advance();
                Ok(Pattern::Literal(AstNode::IntegerLiteral(v)))
            }
            TokenKind::String(v) => {
                let v = v.clone();
                self.advance();
                Ok(Pattern::Literal(AstNode::StringLiteral(v)))
            }
            TokenKind::Bool(v) => {
                let v = *v;
                self.advance();
                Ok(Pattern::Literal(AstNode::BoolLiteral(v)))
            }
            TokenKind::LeftParen => {
                self.advance();
                let mut patterns = Vec::new();
                while !self.check(&TokenKind::RightParen) {
                    patterns.push(self.parse_pattern()?);
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.consume(&TokenKind::RightParen, "Expected ')'")?;
                Ok(Pattern::Tuple(patterns))
            }
            TokenKind::LeftBracket => {
                self.advance();
                let mut patterns = Vec::new();
                while !self.check(&TokenKind::RightBracket) {
                    patterns.push(self.parse_pattern()?);
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.consume(&TokenKind::RightBracket, "Expected ']'")?;
                Ok(Pattern::Array(patterns))
            }
            _ => Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("Unexpected pattern '{}'", self.peek().lexeme),
            }),
        }
    }

    fn parse_tuple_or_grouped(&mut self) -> OmegaResult<AstNode> {
        self.consume(&TokenKind::LeftParen, "Expected '('")?;

        if self.check(&TokenKind::RightParen) {
            self.advance();
            return Ok(AstNode::Tuple(Vec::new()));
        }

        let first = self.parse_expression()?;

        if self.check(&TokenKind::Comma) {
            let mut elements = vec![first];
            while self.check(&TokenKind::Comma) {
                self.advance();
                if self.check(&TokenKind::RightParen) {
                    break;
                }
                elements.push(self.parse_expression()?);
            }
            self.consume(&TokenKind::RightParen, "Expected ')'")?;
            Ok(AstNode::Tuple(elements))
        } else {
            self.consume(&TokenKind::RightParen, "Expected ')'")?;
            Ok(first)
        }
    }

    fn parse_array(&mut self) -> OmegaResult<AstNode> {
        self.consume(&TokenKind::LeftBracket, "Expected '['")?;

        if self.check(&TokenKind::RightBracket) {
            self.advance();
            return Ok(AstNode::Array(Vec::new()));
        }

        let first = self.parse_expression()?;

        // Check for array repeat [value; count]
        if self.check(&TokenKind::Semicolon) {
            self.advance();
            let count = self.parse_expression()?;
            self.consume(&TokenKind::RightBracket, "Expected ']'")?;
            return Ok(AstNode::ArrayRepeat {
                value: Box::new(first),
                count: Box::new(count),
            });
        }

        // Check for comprehension [expr for x in iter]
        if self.check_keyword(Keyword::For) {
            self.advance();
            let variable = self.consume_identifier("Expected variable name")?;
            self.consume_keyword(Keyword::In, "Expected 'in'")?;
            let iter = self.parse_expression()?;
            let condition = if self.check_keyword(Keyword::If) {
                self.advance();
                Some(Box::new(self.parse_expression()?))
            } else {
                None
            };
            self.consume(&TokenKind::RightBracket, "Expected ']'")?;
            return Ok(AstNode::ListComp {
                element: Box::new(first),
                iter: Box::new(iter),
                variable,
                condition,
            });
        }

        let mut elements = vec![first];
        while self.check(&TokenKind::Comma) {
            self.advance();
            if self.check(&TokenKind::RightBracket) {
                break;
            }
            elements.push(self.parse_expression()?);
        }
        self.consume(&TokenKind::RightBracket, "Expected ']'")?;

        Ok(AstNode::Array(elements))
    }

    fn parse_block(&mut self) -> OmegaResult<AstNode> {
        self.consume(&TokenKind::LeftBrace, "Expected '{'")?;
        let mut statements = Vec::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            if self.check(&TokenKind::RightBrace) {
                break;
            }
            statements.push(self.parse_statement()?);
            self.skip_newlines();
        }

        self.consume(&TokenKind::RightBrace, "Expected '}'")?;

        Ok(AstNode::Block(statements))
    }

    fn parse_block_or_expression(&mut self) -> OmegaResult<AstNode> {
        if self.check(&TokenKind::LeftBrace) {
            self.parse_block()
        } else {
            self.parse_expression()
        }
    }

    fn parse_params(&mut self) -> OmegaResult<Vec<Param>> {
        let mut params = Vec::new();

        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            let is_mut = if self.check_keyword(Keyword::Mut) {
                self.advance();
                true
            } else {
                false
            };

            let is_ref = if self.check_keyword(Keyword::Ref) {
                self.advance();
                true
            } else {
                false
            };

            let variadic = if self.check(&TokenKind::DotDotDot) {
                self.advance();
                true
            } else {
                false
            };

            let name = self.consume_identifier("Expected parameter name")?;

            let type_annotation = if self.check(&TokenKind::Colon) {
                self.advance();
                Some(self.parse_type_annotation()?)
            } else {
                None
            };

            let default = if self.check(&TokenKind::Equal) {
                self.advance();
                Some(Box::new(self.parse_expression()?))
            } else {
                None
            };

            params.push(Param {
                name,
                type_annotation,
                default,
                is_mut,
                is_ref,
                variadic,
            });

            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        Ok(params)
    }

    fn parse_call_args(&mut self) -> OmegaResult<Vec<AstNode>> {
        self.consume(&TokenKind::LeftParen, "Expected '('")?;
        let mut args = Vec::new();

        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            args.push(self.parse_expression()?);
            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        self.consume(&TokenKind::RightParen, "Expected ')'")?;
        Ok(args)
    }

    fn parse_type_annotation(&mut self) -> OmegaResult<TypeAnnotation> {
        let kind = match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();

                if name == "Self" {
                    TypeAnnotationKind::SelfType
                } else if name == "_" {
                    TypeAnnotationKind::Infer
                } else if name == "never" {
                    TypeAnnotationKind::Never
                } else if self.check(&TokenKind::Less) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.check(&TokenKind::Greater) {
                        args.push(self.parse_type_annotation()?);
                        if self.check(&TokenKind::Comma) {
                            self.advance();
                        }
                    }
                    self.consume(&TokenKind::Greater, "Expected '>'")?;
                    TypeAnnotationKind::Generic {
                        base: Box::new(TypeAnnotation {
                            kind: TypeAnnotationKind::Simple(name),
                            span: None,
                        }),
                        args,
                    }
                } else {
                    TypeAnnotationKind::Simple(name)
                }
            }
            TokenKind::LeftParen => {
                self.advance();
                let mut types = Vec::new();
                while !self.check(&TokenKind::RightParen) {
                    types.push(self.parse_type_annotation()?);
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.consume(&TokenKind::RightParen, "Expected ')'")?;
                TypeAnnotationKind::Tuple(types)
            }
            TokenKind::LeftBracket => {
                self.advance();
                let element = self.parse_type_annotation()?;
                let size = if self.check(&TokenKind::Semicolon) {
                    self.advance();
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                self.consume(&TokenKind::RightBracket, "Expected ']'")?;
                TypeAnnotationKind::Array { element: Box::new(element), size }
            }
            TokenKind::Ampersand => {
                self.advance();
                let mutable = if self.check_keyword(Keyword::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let inner = self.parse_type_annotation()?;
                TypeAnnotationKind::Reference {
                    mutable,
                    inner: Box::new(inner),
                }
            }
            _ => return Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("Expected type annotation, got '{}'", self.peek().lexeme),
            }),
        };

        // Handle optional type
        let kind = if self.check(&TokenKind::Question) {
            self.advance();
            TypeAnnotationKind::Optional(Box::new(TypeAnnotation { kind, span: None }))
        } else {
            kind
        };

        // Handle function type
        let kind = if self.check(&TokenKind::Arrow) {
            self.advance();
            let return_type = self.parse_type_annotation()?;
            match kind {
                TypeAnnotationKind::Tuple(params) => TypeAnnotationKind::Function {
                    params,
                    return_type: Box::new(return_type),
                },
                _ => TypeAnnotationKind::Function {
                    params: vec![TypeAnnotation { kind, span: None }],
                    return_type: Box::new(return_type),
                },
            }
        } else {
            kind
        };

        Ok(TypeAnnotation { kind, span: None })
    }

    fn parse_type_params(&mut self) -> OmegaResult<Vec<TypeParam>> {
        self.consume(&TokenKind::Less, "Expected '<'")?;
        let mut params = Vec::new();

        while !self.check(&TokenKind::Greater) && !self.is_at_end() {
            let name = self.consume_identifier("Expected type parameter name")?;

            let mut bounds = Vec::new();
            if self.check(&TokenKind::Colon) {
                self.advance();
                loop {
                    bounds.push(self.consume_identifier("Expected trait bound")?);
                    if !self.check(&TokenKind::Plus) {
                        break;
                    }
                    self.advance();
                }
            }

            let default = if self.check(&TokenKind::Equal) {
                self.advance();
                Some(self.parse_type_annotation()?)
            } else {
                None
            };

            params.push(TypeParam { name, bounds, default });

            if self.check(&TokenKind::Comma) {
                self.advance();
            }
        }

        self.consume(&TokenKind::Greater, "Expected '>'")?;
        Ok(params)
    }

    fn check_compound_assignment(&self) -> bool {
        matches!(self.peek().kind,
            TokenKind::PlusEqual | TokenKind::MinusEqual | TokenKind::StarEqual |
            TokenKind::SlashEqual | TokenKind::PercentEqual |
            TokenKind::AmpersandEqual | TokenKind::PipeEqual | TokenKind::CaretEqual |
            TokenKind::LessLessEqual | TokenKind::GreaterGreaterEqual | TokenKind::StarStarEqual
        )
    }

    fn get_compound_op(&self) -> BinaryOp {
        match &self.peek().kind {
            TokenKind::PlusEqual => BinaryOp::Add,
            TokenKind::MinusEqual => BinaryOp::Sub,
            TokenKind::StarEqual => BinaryOp::Mul,
            TokenKind::SlashEqual => BinaryOp::Div,
            TokenKind::PercentEqual => BinaryOp::Mod,
            TokenKind::AmpersandEqual => BinaryOp::BitAnd,
            TokenKind::PipeEqual => BinaryOp::BitOr,
            TokenKind::CaretEqual => BinaryOp::BitXor,
            TokenKind::LessLessEqual => BinaryOp::Shl,
            TokenKind::GreaterGreaterEqual => BinaryOp::Shr,
            TokenKind::StarStarEqual => BinaryOp::Pow,
            _ => BinaryOp::Add,
        }
    }

    // Helper methods
    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token {
            kind: TokenKind::Eof,
            lexeme: String::new(),
            line: 0,
            col: 0,
            offset: 0,
        })
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        &self.tokens[self.current - 1]
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    fn check_keyword(&self, keyword: Keyword) -> bool {
        matches!(&self.peek().kind, TokenKind::Keyword(k) if *k == keyword)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn consume(&mut self, kind: &TokenKind, message: &str) -> OmegaResult<&Token> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("{} (got '{}')", message, self.peek().lexeme),
            })
        }
    }

    fn consume_identifier(&mut self, message: &str) -> OmegaResult<String> {
        match &self.peek().kind.clone() {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("{} (got '{}')", message, self.peek().lexeme),
            }),
        }
    }

    fn consume_keyword(&mut self, keyword: Keyword, message: &str) -> OmegaResult<&Token> {
        if self.check_keyword(keyword) {
            Ok(self.advance())
        } else {
            Err(OmegaError::ParseError {
                location: format!("{}:{}", self.peek().line, self.peek().col),
                message: format!("{} (got '{}')", message, self.peek().lexeme),
            })
        }
    }

    fn skip_newlines(&mut self) {
        while self.check(&TokenKind::Newline) || self.check(&TokenKind::Indent) || self.check(&TokenKind::Dedent) {
            self.advance();
        }
    }

    fn synchronize(&mut self) {
        while !self.is_at_end() {
            if self.check(&TokenKind::Semicolon) {
                self.advance();
                return;
            }
            match &self.peek().kind {
                TokenKind::Keyword(k) if k.is_declaration() || k.is_control_flow() => return,
                _ => {}
            }
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_let() {
        let mut parser = Parser::new("let x = 42");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                match &stmts[0] {
                    AstNode::LetBinding { name, value, .. } => {
                        assert_eq!(name, "x");
                        assert!(value.is_some());
                    }
                    _ => panic!("Expected LetBinding"),
                }
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_function() {
        let mut parser = Parser::new("fn add(a, b) { return a + b }");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                match &stmts[0] {
                    AstNode::FunctionDef { name, params, .. } => {
                        assert_eq!(name, "add");
                        assert_eq!(params.len(), 2);
                    }
                    _ => panic!("Expected FunctionDef"),
                }
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_if() {
        let mut parser = Parser::new("if x > 0 { print(x) }");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                assert!(matches!(stmts[0], AstNode::If { .. }));
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_while() {
        let mut parser = Parser::new("while x < 10 { x += 1 }");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                assert!(matches!(stmts[0], AstNode::While { .. }));
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_for() {
        let mut parser = Parser::new("for i in 0..10 { print(i) }");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                assert!(matches!(stmts[0], AstNode::For { .. }));
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_binary_ops() {
        let mut parser = Parser::new("let x = 1 + 2 * 3");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                match &stmts[0] {
                    AstNode::LetBinding { value, .. } => {
                        match value.as_ref().unwrap().as_ref() {
                            AstNode::BinaryOp { op, .. } => assert_eq!(*op, BinaryOp::Add),
                            _ => panic!("Expected BinaryOp"),
                        }
                    }
                    _ => panic!("Expected LetBinding"),
                }
            }
            _ => panic!("Expected Program"),
        }
    }

    #[test]
    fn test_parse_array() {
        let mut parser = Parser::new("let arr = [1, 2, 3]");
        let ast = parser.parse().unwrap();
        match ast {
            AstNode::Program(stmts) => {
                assert_eq!(stmts.len(), 1);
                match &stmts[0] {
                    AstNode::LetBinding { value, .. } => {
                        match value.as_ref().unwrap().as_ref() {
                            AstNode::Array(elems) => assert_eq!(elems.len(), 3),
                            _ => panic!("Expected Array"),
                        }
                    }
                    _ => panic!("Expected LetBinding"),
                }
            }
            _ => panic!("Expected Program"),
        }
    }
}
