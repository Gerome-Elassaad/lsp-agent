use tree_sitter::{Language, Parser, Query, QueryCursor};

pub struct RustCodeParser {
    language: Language,
}

impl RustCodeParser {
    pub fn new() -> Self {
        let language = tree_sitter_rust::language();
        RustCodeParser { language }
    }

    pub fn parse(&self, code: &str) -> Option<tree_sitter::Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.language).unwrap();
        parser.parse(code, None)
    }

    pub fn extract_function_definitions(&self, code: &str) -> Vec<String> {
        let tree = self.parse(code).unwrap();
        let query = Query::new(
            self.language,
            "(function_item name: (identifier) @name)",
        ).unwrap();

        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(&query, tree.root_node(), code.as_bytes());

        let mut function_names = Vec::new();
        for m in matches {
            for capture in m.captures {
                if query.capture_names()[capture.index as usize] == "name" {
                    function_names.push(capture.node.utf8_text(code.as_bytes()).unwrap().to_string());
                }
            }
        }

        function_names
    }
}
