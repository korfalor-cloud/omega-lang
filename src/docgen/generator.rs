use std::collections::HashMap;
use crate::ast::*;
use crate::errors::OmegaResult;

pub struct DocGenerator {
    output_format: DocFormat,
    include_private: bool,
    include_examples: bool,
    base_url: String,
    docs: Vec<DocItem>,
}

#[derive(Debug, Clone)]
pub enum DocFormat {
    Markdown,
    Html,
    Json,
}

#[derive(Debug, Clone)]
pub enum DocItem {
    Module(DocModule),
    Function(DocFunction),
    Struct(DocStruct),
    Enum(DocEnum),
    Trait(DocTrait),
    Constant(DocConstant),
    TypeAlias(DocTypeAlias),
}

#[derive(Debug, Clone)]
pub struct DocModule {
    pub name: String,
    pub description: String,
    pub items: Vec<DocItem>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocFunction {
    pub name: String,
    pub description: String,
    pub params: Vec<DocParam>,
    pub return_type: Option<String>,
    pub is_async: bool,
    pub is_pub: bool,
    pub examples: Vec<String>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocParam {
    pub name: String,
    pub type_name: Option<String>,
    pub description: String,
    pub default: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocStruct {
    pub name: String,
    pub description: String,
    pub fields: Vec<DocField>,
    pub methods: Vec<DocFunction>,
    pub is_pub: bool,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocField {
    pub name: String,
    pub type_name: String,
    pub description: String,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct DocEnum {
    pub name: String,
    pub description: String,
    pub variants: Vec<DocVariant>,
    pub is_pub: bool,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocVariant {
    pub name: String,
    pub description: String,
    pub fields: Vec<DocField>,
}

#[derive(Debug, Clone)]
pub struct DocTrait {
    pub name: String,
    pub description: String,
    pub methods: Vec<DocFunction>,
    pub is_pub: bool,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocConstant {
    pub name: String,
    pub type_name: String,
    pub value: String,
    pub description: String,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct DocTypeAlias {
    pub name: String,
    pub target: String,
    pub description: String,
    pub line: usize,
}

impl DocGenerator {
    pub fn new() -> Self {
        Self {
            output_format: DocFormat::Markdown,
            include_private: false,
            include_examples: true,
            base_url: String::new(),
            docs: Vec::new(),
        }
    }

    pub fn with_format(mut self, format: DocFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn with_private(mut self, include: bool) -> Self {
        self.include_private = include;
        self
    }

    pub fn with_examples(mut self, include: bool) -> Self {
        self.include_examples = include;
        self
    }

    pub fn generate(&mut self, ast: &AstNode) -> OmegaResult<String> {
        self.docs.clear();
        self.visit_node(ast)?;

        match self.output_format {
            DocFormat::Markdown => self.generate_markdown(),
            DocFormat::Html => self.generate_html(),
            DocFormat::Json => self.generate_json(),
        }
    }

    fn visit_node(&mut self, node: &AstNode) -> OmegaResult<()> {
        match node {
            AstNode::Program(stmts) => {
                for stmt in stmts {
                    self.visit_node(stmt)?;
                }
            }
            AstNode::Module { name, body, .. } => {
                let module = DocModule {
                    name: name.clone(),
                    description: String::new(),
                    items: Vec::new(),
                    line: 0,
                };
                self.docs.push(DocItem::Module(module));
            }
            AstNode::FunctionDef {
                name,
                params,
                return_type,
                is_async,
                is_pub,
                body,
                ..
            } => {
                if !is_pub && !self.include_private {
                    return Ok(());
                }

                let doc_params: Vec<DocParam> = params
                    .iter()
                    .map(|p| DocParam {
                        name: p.name.clone(),
                        type_name: p.type_annotation.as_ref().map(|t| format!("{:?}", t)),
                        description: String::new(),
                        default: None,
                    })
                    .collect();

                let func = DocFunction {
                    name: name.clone(),
                    description: String::new(),
                    params: doc_params,
                    return_type: return_type.as_ref().map(|t| format!("{:?}", t)),
                    is_async: *is_async,
                    is_pub: *is_pub,
                    examples: Vec::new(),
                    line: 0,
                };
                self.docs.push(DocItem::Function(func));
            }
            AstNode::StructDef {
                name,
                fields,
                is_pub,
                ..
            } => {
                if !is_pub && !self.include_private {
                    return Ok(());
                }

                let doc_fields: Vec<DocField> = fields
                    .iter()
                    .map(|f| DocField {
                        name: f.name.clone(),
                        type_name: format!("{:?}", f.type_annotation),
                        description: String::new(),
                        is_pub: f.is_pub,
                    })
                    .collect();

                let doc_struct = DocStruct {
                    name: name.clone(),
                    description: String::new(),
                    fields: doc_fields,
                    methods: Vec::new(),
                    is_pub: *is_pub,
                    line: 0,
                };
                self.docs.push(DocItem::Struct(doc_struct));
            }
            AstNode::EnumDef {
                name,
                variants,
                is_pub,
                ..
            } => {
                if !is_pub && !self.include_private {
                    return Ok(());
                }

                let doc_variants: Vec<DocVariant> = variants
                    .iter()
                    .map(|v| DocVariant {
                        name: v.name.clone(),
                        description: String::new(),
                        fields: Vec::new(),
                    })
                    .collect();

                let doc_enum = DocEnum {
                    name: name.clone(),
                    description: String::new(),
                    variants: doc_variants,
                    is_pub: *is_pub,
                    line: 0,
                };
                self.docs.push(DocItem::Enum(doc_enum));
            }
            AstNode::TraitDef {
                name,
                items,
                is_pub,
                ..
            } => {
                if !is_pub && !self.include_private {
                    return Ok(());
                }

                let doc_trait = DocTrait {
                    name: name.clone(),
                    description: String::new(),
                    methods: Vec::new(),
                    is_pub: *is_pub,
                    line: 0,
                };
                self.docs.push(DocItem::Trait(doc_trait));
            }
            AstNode::ConstBinding { name, type_annotation, value, .. } => {
                let doc_const = DocConstant {
                    name: name.clone(),
                    type_name: type_annotation.as_ref().map(|t| format!("{:?}", t)).unwrap_or_default(),
                    value: format!("{:?}", value),
                    description: String::new(),
                    line: 0,
                };
                self.docs.push(DocItem::Constant(doc_const));
            }
            _ => {}
        }
        Ok(())
    }

    fn generate_markdown(&self) -> OmegaResult<String> {
        let mut output = String::new();
        output.push_str("# Omega Documentation\n\n");
        output.push_str("## Table of Contents\n\n");

        // Generate TOC
        for item in &self.docs {
            match item {
                DocItem::Module(m) => {
                    output.push_str(&format!("- [Module: {}](#{})\n", m.name, m.name.to_lowercase()));
                }
                DocItem::Function(f) => {
                    output.push_str(&format!("- [Function: {}](#{})\n", f.name, f.name.to_lowercase()));
                }
                DocItem::Struct(s) => {
                    output.push_str(&format!("- [Struct: {}](#{})\n", s.name, s.name.to_lowercase()));
                }
                DocItem::Enum(e) => {
                    output.push_str(&format!("- [Enum: {}](#{})\n", e.name, e.name.to_lowercase()));
                }
                DocItem::Trait(t) => {
                    output.push_str(&format!("- [Trait: {}](#{})\n", t.name, t.name.to_lowercase()));
                }
                _ => {}
            }
        }

        output.push_str("\n---\n\n");

        // Generate documentation for each item
        for item in &self.docs {
            match item {
                DocItem::Module(m) => {
                    output.push_str(&format!("## Module: {}\n\n", m.name));
                    if !m.description.is_empty() {
                        output.push_str(&format!("{}\n\n", m.description));
                    }
                }
                DocItem::Function(f) => {
                    output.push_str(&format!("## Function: {}\n\n", f.name));
                    if !f.description.is_empty() {
                        output.push_str(&format!("{}\n\n", f.description));
                    }

                    output.push_str("### Signature\n\n");
                    output.push_str("```omega\n");
                    if f.is_async {
                        output.push_str("async ");
                    }
                    if f.is_pub {
                        output.push_str("pub ");
                    }
                    output.push_str(&format!("fn {}(", f.name));
                    for (i, param) in f.params.iter().enumerate() {
                        if i > 0 {
                            output.push_str(", ");
                        }
                        output.push_str(&param.name);
                        if let Some(ty) = &param.type_name {
                            output.push_str(&format!(": {}", ty));
                        }
                    }
                    output.push_str(")");
                    if let Some(ret) = &f.return_type {
                        output.push_str(&format!(" -> {}", ret));
                    }
                    output.push_str("\n```\n\n");

                    if !f.params.is_empty() {
                        output.push_str("### Parameters\n\n");
                        output.push_str("| Name | Type | Description |\n");
                        output.push_str("|------|------|-------------|\n");
                        for param in &f.params {
                            output.push_str(&format!(
                                "| {} | {} | {} |\n",
                                param.name,
                                param.type_name.as_deref().unwrap_or("Any"),
                                param.description
                            ));
                        }
                        output.push_str("\n");
                    }
                }
                DocItem::Struct(s) => {
                    output.push_str(&format!("## Struct: {}\n\n", s.name));
                    if !s.description.is_empty() {
                        output.push_str(&format!("{}\n\n", s.description));
                    }

                    if !s.fields.is_empty() {
                        output.push_str("### Fields\n\n");
                        output.push_str("| Name | Type | Description |\n");
                        output.push_str("|------|------|-------------|\n");
                        for field in &s.fields {
                            output.push_str(&format!(
                                "| {} | {} | {} |\n",
                                field.name, field.type_name, field.description
                            ));
                        }
                        output.push_str("\n");
                    }
                }
                DocItem::Enum(e) => {
                    output.push_str(&format!("## Enum: {}\n\n", e.name));
                    if !e.description.is_empty() {
                        output.push_str(&format!("{}\n\n", e.description));
                    }

                    if !e.variants.is_empty() {
                        output.push_str("### Variants\n\n");
                        for variant in &e.variants {
                            output.push_str(&format!("- **{}**: {}\n", variant.name, variant.description));
                        }
                        output.push_str("\n");
                    }
                }
                DocItem::Trait(t) => {
                    output.push_str(&format!("## Trait: {}\n\n", t.name));
                    if !t.description.is_empty() {
                        output.push_str(&format!("{}\n\n", t.description));
                    }

                    if !t.methods.is_empty() {
                        output.push_str("### Methods\n\n");
                        for method in &t.methods {
                            output.push_str(&format!("- {}\n", method.name));
                        }
                        output.push_str("\n");
                    }
                }
                DocItem::Constant(c) => {
                    output.push_str(&format!("## Constant: {}\n\n", c.name));
                    output.push_str(&format!("- Type: {}\n", c.type_name));
                    output.push_str(&format!("- Value: `{}`\n", c.value));
                    if !c.description.is_empty() {
                        output.push_str(&format!("- {}\n", c.description));
                    }
                    output.push_str("\n");
                }
                DocItem::TypeAlias(t) => {
                    output.push_str(&format!("## Type Alias: {}\n\n", t.name));
                    output.push_str(&format!("- Target: `{}`\n", t.target));
                    if !t.description.is_empty() {
                        output.push_str(&format!("- {}\n", t.description));
                    }
                    output.push_str("\n");
                }
            }
        }

        Ok(output)
    }

    fn generate_html(&self) -> OmegaResult<String> {
        let mut output = String::new();
        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("<title>Omega Documentation</title>\n");
        output.push_str("<style>\n");
        output.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }\n");
        output.push_str("h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }\n");
        output.push_str("h2 { color: #555; }\n");
        output.push_str("pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }\n");
        output.push_str("code { font-family: 'Fira Code', 'Consolas', monospace; }\n");
        output.push_str("table { border-collapse: collapse; width: 100%; margin: 10px 0; }\n");
        output.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        output.push_str("th { background: #f5f5f5; }\n");
        output.push_str(".function { background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }\n");
        output.push_str(".struct { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }\n");
        output.push_str(".enum { background: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0; }\n");
        output.push_str("</style>\n</head>\n<body>\n");
        output.push_str("<h1>Omega Documentation</h1>\n");

        for item in &self.docs {
            match item {
                DocItem::Function(f) => {
                    output.push_str(&format!("<div class='function'>\n"));
                    output.push_str(&format!("<h2>{}</h2>\n", f.name));
                    if !f.description.is_empty() {
                        output.push_str(&format!("<p>{}</p>\n", f.description));
                    }
                    output.push_str("</div>\n");
                }
                DocItem::Struct(s) => {
                    output.push_str(&format!("<div class='struct'>\n"));
                    output.push_str(&format!("<h2>{}</h2>\n", s.name));
                    if !s.description.is_empty() {
                        output.push_str(&format!("<p>{}</p>\n", s.description));
                    }
                    output.push_str("</div>\n");
                }
                _ => {}
            }
        }

        output.push_str("</body>\n</html>");
        Ok(output)
    }

    fn generate_json(&self) -> OmegaResult<String> {
        let mut output = String::new();
        output.push_str("{\n");
        output.push_str("  \"items\": [\n");

        for (i, item) in self.docs.iter().enumerate() {
            if i > 0 {
                output.push_str(",\n");
            }
            match item {
                DocItem::Function(f) => {
                    output.push_str(&format!(
                        "    {{ \"type\": \"function\", \"name\": \"{}\" }}",
                        f.name
                    ));
                }
                DocItem::Struct(s) => {
                    output.push_str(&format!(
                        "    {{ \"type\": \"struct\", \"name\": \"{}\" }}",
                        s.name
                    ));
                }
                DocItem::Enum(e) => {
                    output.push_str(&format!(
                        "    {{ \"type\": \"enum\", \"name\": \"{}\" }}",
                        e.name
                    ));
                }
                _ => {}
            }
        }

        output.push_str("\n  ]\n}");
        Ok(output)
    }
}

// Documentation parser for extracting doc comments
pub struct DocCommentParser {
    comments: Vec<DocComment>,
}

#[derive(Debug, Clone)]
pub struct DocComment {
    pub text: String,
    pub line: usize,
    pub is_doc_comment: bool,
}

impl DocCommentParser {
    pub fn new() -> Self {
        Self {
            comments: Vec::new(),
        }
    }

    pub fn parse(source: &str) -> Vec<DocComment> {
        let mut comments = Vec::new();
        let mut in_block_comment = false;
        let mut current_comment = String::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("///") {
                let text = trimmed[3..].trim().to_string();
                comments.push(DocComment {
                    text,
                    line: line_num + 1,
                    is_doc_comment: true,
                });
            } else if trimmed.starts_with("/**") {
                in_block_comment = true;
                current_comment.clear();
                let text = trimmed[3..].trim().to_string();
                if !text.is_empty() {
                    current_comment.push_str(&text);
                }
            } else if in_block_comment {
                if trimmed.ends_with("*/") {
                    in_block_comment = false;
                    let text = trimmed[..trimmed.len() - 2].trim().to_string();
                    if !text.is_empty() {
                        current_comment.push(' ');
                        current_comment.push_str(&text);
                    }
                    comments.push(DocComment {
                        text: current_comment.clone(),
                        line: line_num + 1,
                        is_doc_comment: true,
                    });
                } else {
                    if !current_comment.is_empty() {
                        current_comment.push(' ');
                    }
                    current_comment.push_str(trimmed);
                }
            } else if trimmed.starts_with("//") {
                let text = trimmed[2..].trim().to_string();
                comments.push(DocComment {
                    text,
                    line: line_num + 1,
                    is_doc_comment: false,
                });
            }
        }

        comments
    }

    pub fn get_doc_for_line(&self, line: usize) -> Option<&DocComment> {
        self.comments
            .iter()
            .filter(|c| c.is_doc_comment)
            .find(|c| c.line == line - 1 || c.line == line - 2)
    }
}
