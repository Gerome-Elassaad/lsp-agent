use tree_sitter::{Language, Parser, Query, QueryCursor};

pub struct JavascriptCodeParser {
    language: Language,
}

impl JavascriptCodeParser {
    pub fn new() -> Self {
        let language = tree_sitter_javascript::language();
        JavascriptCodeParser { language }
    }

    pub fn parse(&self, code: &str) -> Option<tree_sitter::Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.language).unwrap();
        parser.parse(code, None)
    }
}
