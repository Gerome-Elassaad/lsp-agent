use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub(crate) type Kwargs = HashMap<String, Value>;

const fn max_requests_per_second_default() -> f32 {
    1.
}

const fn true_default() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PostProcess {
    pub(crate) extractor: Option<String>,
    #[serde(default = "true_default")]
    pub(crate) remove_duplicate_start: bool,
    #[serde(default = "true_default")]
    pub(crate) remove_duplicate_end: bool,
}

impl Default for PostProcess {
    fn default() -> Self {
        Self {
            extractor: None,
            remove_duplicate_start: true,
            remove_duplicate_end: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ValidSplitter {
    #[serde(rename = "tree_sitter")]
    TreeSitter(TreeSitter),
    #[serde(rename = "text_splitter")]
    TextSplitter(TextSplitter),
}

impl Default for ValidSplitter {
    fn default() -> Self {
        ValidSplitter::TreeSitter(TreeSitter::default())
    }
}

const fn chunk_size_default() -> usize {
    1500
}

const fn chunk_overlap_default() -> usize {
    0
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TreeSitter {
    #[serde(default = "chunk_size_default")]
    pub(crate) chunk_size: usize,
    #[serde(default = "chunk_overlap_default")]
    pub(crate) chunk_overlap: usize,
}

impl Default for TreeSitter {
    fn default() -> Self {
        Self {
            chunk_size: 1500,
            chunk_overlap: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextSplitter {
    #[serde(default = "chunk_size_default")]
    pub(crate) chunk_size: usize,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct EmbeddingPrefix {
    #[serde(default)]
    pub(crate) storage: String,
    #[serde(default)]
    pub(crate) retrieval: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OllamaEmbeddingModel {
    // The generate endpoint, default: 'http://localhost:11434/api/embeddings'
    pub(crate) endpoint: Option<String>,
    // The model name
    pub(crate) model: String,
    // The prefix to apply to the embeddings
    #[serde(default)]
    pub(crate) prefix: EmbeddingPrefix,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ValidEmbeddingModel {
    #[serde(rename = "ollama")]
    Ollama(OllamaEmbeddingModel),
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum VectorDataType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "binary")]
    Binary,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VectorStore {
    pub(crate) crawl: Option<Crawl>,
    #[serde(default)]
    pub(crate) splitter: ValidSplitter,
    pub(crate) embedding_model: ValidEmbeddingModel,
    pub(crate) data_type: VectorDataType,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum ValidMemoryBackend {
    #[serde(rename = "file_store")]
    FileStore(FileStore),
    #[serde(rename = "vector_store")]
    VectorStore(VectorStore),
    #[serde(rename = "postgresml")]
    PostgresML(PostgresML),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ValidModel {
    #[cfg(feature = "llama_cpp")]
    #[serde(rename = "llama_cpp")]
    LLaMACPP(LLaMACPP),
    #[serde(rename = "open_ai")]
    OpenAI(OpenAI),
    #[serde(rename = "anthropic")]
    Anthropic(Anthropic),
    #[serde(rename = "mistral_fim")]
    MistralFIM(MistralFIM),
    #[serde(rename = "ollama")]
    Ollama(Ollama),
    #[serde(rename = "gemini")]
    Gemini(Gemini),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

impl ChatMessage {
    pub(crate) fn new(role: String, content: String) -> Self {
        Self {
            role,
            content,
            // tool_calls: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
#[serde(deny_unknown_fields)]
pub(crate) struct FIM {
    pub(crate) start: String,
    pub(crate) middle: String,
    pub(crate) end: String,
}

const fn max_crawl_memory_default() -> u64 {
    100_000_000
}

const fn max_crawl_file_size_default() -> u64 {
    10_000_000
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Crawl {
    #[serde(default = "max_crawl_file_size_default")]
    pub(crate) max_file_size: u64,
    #[serde(default = "max_crawl_memory_default")]
    pub(crate) max_crawl_memory: u64,
    #[serde(default)]
    pub(crate) all_files: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct PostgresMLEmbeddingModel {
    pub(crate) model: String,
    pub(crate) embed_parameters: Option<Value>,
    pub(crate) query_parameters: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PostgresML {
    pub(crate) database_url: Option<String>,
    pub(crate) crawl: Option<Crawl>,
    #[serde(default)]
    pub(crate) splitter: ValidSplitter,
    pub(crate) embedding_model: Option<PostgresMLEmbeddingModel>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileStore {
    pub(crate) crawl: Option<Crawl>,
}

impl FileStore {
    pub(crate) fn new_without_crawl() -> Self {
        Self { crawl: None }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Ollama {
    // The generate endpoint, default: 'http://localhost:11434/api/generate'
    pub(crate) generate_endpoint: Option<String>,
    // The chat endpoint, default: 'http://localhost:11434/api/chat'
    pub(crate) chat_endpoint: Option<String>,
    // The model name
    pub(crate) model: String,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MistralFIM {
    // The auth token env var name
    pub(crate) auth_token_env_var_name: Option<String>,
    pub(crate) auth_token: Option<String>,
    // The fim endpoint
    pub(crate) fim_endpoint: Option<String>,
    // The model name
    pub(crate) model: String,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
}

#[cfg(feature = "llama_cpp")]
const fn n_gpu_layers_default() -> u32 {
    1000
}

#[cfg(feature = "llama_cpp")]
const fn n_ctx_default() -> u32 {
    1000
}

#[cfg(feature = "llama_cpp")]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LLaMACPP {
    // Which model to use
    pub(crate) repository: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) file_path: Option<String>,
    // The layers to put on the GPU
    #[serde(default = "n_gpu_layers_default")]
    pub(crate) n_gpu_layers: u32,
    // The context size
    #[serde(default = "n_ctx_default")]
    pub(crate) n_ctx: u32,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAI {
    // The auth token env var name
    pub(crate) auth_token_env_var_name: Option<String>,
    // The auth token
    pub(crate) auth_token: Option<String>,
    // The completions endpoint
    pub(crate) completions_endpoint: Option<String>,
    // The chat endpoint
    pub(crate) chat_endpoint: Option<String>,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
    // The model name
    pub(crate) model: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Gemini {
    // The auth token env var name
    pub(crate) auth_token_env_var_name: Option<String>,
    // The auth token
    pub(crate) auth_token: Option<String>,
    // The completions endpoint
    #[allow(dead_code)]
    pub(crate) completions_endpoint: Option<String>,
    // The chat endpoint
    pub(crate) chat_endpoint: Option<String>,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
    // The model name
    pub(crate) model: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Anthropic {
    // The auth token env var name
    pub(crate) auth_token_env_var_name: Option<String>,
    // The auth token
    pub(crate) auth_token: Option<String>,
    // The completions endpoint
    #[allow(dead_code)]
    pub(crate) completions_endpoint: Option<String>,
    // The chat endpoint
    pub(crate) chat_endpoint: Option<String>,
    // The maximum requests per second
    #[serde(default = "max_requests_per_second_default")]
    pub(crate) max_requests_per_second: f32,
    // The model name
    pub(crate) model: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Completion {
    // The model key to use
    pub(crate) model: String,
    // Args are deserialized by the backend using them
    #[serde(default)]
    pub(crate) parameters: Kwargs,
    // Parameters for post processing
    #[serde(default)]
    pub(crate) post_process: PostProcess,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Chat {
    // The trigger text
    pub(crate) trigger: String,
    // The name to display in the editor
    pub(crate) action_display_name: String,
    // The model key to use
    pub(crate) model: String,
    // Args are deserialized by the backend using them
    #[serde(default)]
    pub(crate) parameters: Kwargs,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Action {
    // The name to display in the editor
    pub(crate) action_display_name: String,
    // The model key to use
    pub(crate) model: String,
    // Args are deserialized by the backend using them
    #[serde(default)]
    pub(crate) parameters: Kwargs,
    // Parameters for post processing
    #[serde(default)]
    pub(crate) post_process: PostProcess,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelSelectionConfig {
    pub(crate) default_model: String,
    #[serde(default)]
    pub(crate) request_type_model: HashMap<String, String>,
    #[serde(default)]
    pub(crate) language_model: HashMap<String, String>,
    #[serde(default)]
    pub(crate) file_type_model: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidConfig {
    pub(crate) memory: ValidMemoryBackend,
    pub(crate) models: HashMap<String, ValidModel>,
    pub(crate) completion: Option<Completion>,
    #[serde(default)]
    pub(crate) actions: Vec<Action>,
    #[serde(default)]
    #[serde(alias = "chat")] // Legacy from when it was called chat, remove soon
    pub(crate) chats: Vec<Chat>,
    #[serde(default)]
    pub(crate) model_selection: ModelSelectionConfig,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct ValidClientParams {
    #[serde(alias = "rootUri")]
    pub(crate) root_uri: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) config: ValidConfig,
    pub(crate) client_params: ValidClientParams,
}

impl Config {
    pub(crate) fn new(mut args: Value) -> Result<Self> {
        // Validate that the models specified are there so we can unwrap
        let configuration_args = args
            .as_object_mut()
            .context("Server configuration must be a JSON object")?
            .remove("initializationOptions");
        let valid_args = match configuration_args {
            Some(configuration_args) => serde_json::from_value(configuration_args)?,
            None => anyhow::bail!("lsp-ai does not currently provide a default configuration. Please pass a configuration. See https://github.com/Gerome-Elassaad/lsp-ai for configuration options and examples"),
        };
        let client_params: ValidClientParams = serde_json::from_value(args)?;
        Ok(Self {
            config: valid_args,
            client_params,
        })
    }

    ///////////////////////////////////////
    // Helpers for the backends ///////////
    ///////////////////////////////////////

    pub(crate) fn get_chats(&self) -> &Vec<Chat> {
        &self.config.chats
    }

    pub(crate) fn get_actions(&self) -> &Vec<Action> {
        &self.config.actions
    }

    pub(crate) fn get_completions_post_process(&self) -> Option<&PostProcess> {
        self.config.completion.as_ref().map(|x| &x.post_process)
    }

    pub(crate) fn get_completion_transformer_max_requests_per_second(&self) -> anyhow::Result<f32> {
        match &self
            .config
            .models
            .get(
                &self
                    .config
                    .completion
                    .as_ref()
                    .context("Completions is not enabled")?
                    .model,
            )
            .with_context(|| {
                format!(
                    "`{}` model not found in `models` config",
                    &self.config.completion.as_ref().unwrap().model
                )
            })? {
            #[cfg(feature = "llama_cpp")]
            ValidModel::LLaMACPP(llama_cpp) => Ok(llama_cpp.max_requests_per_second),
            ValidModel::OpenAI(open_ai) => Ok(open_ai.max_requests_per_second),
            ValidModel::Gemini(gemini) => Ok(gemini.max_requests_per_second),
            ValidModel::Anthropic(anthropic) => Ok(anthropic.max_requests_per_second),
            ValidModel::MistralFIM(mistral_fim) => Ok(mistral_fim.max_requests_per_second),
            ValidModel::Ollama(ollama) => Ok(ollama.max_requests_per_second),
        }
    }
}

// For teesting use only
#[cfg(test)]
impl Config {
    pub(crate) fn default_with_file_store_without_models() -> Self {
        Self {
            config: ValidConfig {
                memory: ValidMemoryBackend::FileStore(FileStore { crawl: None }),
                models: HashMap::new(),
                completion: None,
                actions: vec![],
                chats: vec![],
                model_selection: ModelSelectionConfig {
                    default_model: "default".to_string(),
                    request_type_model: HashMap::new(),
                    language_model: HashMap::new(),
                    file_type_model: HashMap::new(),
                },
            },
            client_params: ValidClientParams { root_uri: None },
        }
    }

    pub(crate) fn default_with_vector_store(vector_store: VectorStore) -> Self {
        Self {
            config: ValidConfig {
                memory: ValidMemoryBackend::VectorStore(vector_store),
                models: HashMap::new(),
                completion: None,
                actions: vec![],
                chats: vec![],
                model_selection: ModelSelectionConfig {
                    default_model: "default".to_string(),
                    request_type_model: HashMap::new(),
                    language_model: HashMap::new(),
                    file_type_model: HashMap::new(),
                },
            },
            client_params: ValidClientParams { root_uri: None },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;

    #[test]
    #[cfg(feature = "llama_cpp")]
    fn llama_cpp_config() {
        let args = json!({
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "model1": {
                        "type": "llama_cpp",
                        "repository": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
                        "name": "deepseek-coder-6.7b-instruct.Q5_K_S.gguf",
                        "n_ctx": 2048,
                        "n_gpu_layers": 35
                    }
                },
                "completion": {
                    "model": "model1",
                    "parameters": {
                        "fim": {
                            "start": "<fim_prefix>",
                            "middle": "<fim_suffix>",
                            "end": "<fim_middle>"
                        },
                        "max_context": 1024,
                        "max_new_tokens": 32,
                    }
                }
            }
        });
        Config::new(args).unwrap();
    }

    #[test]
    fn ollama_config() {
        let args = json!({
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "model1": {
                        "type": "ollama",
                        "model": "llama3"
                    }
                },
                "completion": {
                    "model": "model1",
                    "parameters": {
                        "max_context": 1024,
                        "options": {
                            "num_predict": 32
                        }
                    },
                    "post_process": {
                        "remove_duplicate_start": true,
                        "remove_duplicate_end": true,
                    }
                }
            }
        });
        Config::new(args).unwrap();
    }

    #[test]
    fn open_ai_config() {
        let args = json!({
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "model1": {
                        "type": "open_ai",
                        "completions_endpoint": "https://api.fireworks.ai/inference/v1/completions",
                        "model": "accounts/fireworks/models/llama-v2-34b-code",
                        "auth_token_env_var_name": "FIREWORKS_API_KEY",
                    },
                },
                "completion": {
                    "model": "model1",
                    "parameters": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "Test",
                            },
                            {
                                "role": "user",
                                "content": "Test {CONTEXT} - {CODE}"
                            }
                        ],
                        "max_new_tokens": 32,
                    }
                }
            }
        });
        Config::new(args).unwrap();
    }

    #[test]
    fn gemini_config() {
        let args = json!({
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "model1": {
                        "type": "gemini",
                        "completions_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/",
                        "model": "gemini-1.5-flash-latest",
                        "auth_token_env_var_name": "GEMINI_API_KEY",
                    },
                },
                "completion": {
                    "model": "model1",
                    "parameters": {
                        "systemInstruction": {
                            "role": "system",
                            "parts": [{
                                "text": "TEST system instruction"
                            }]
                        },
                        "generationConfig": {
                            "maxOutputTokens": 10
                        },
                        "contents": [
                          {
                            "role": "user",
                            "parts":[{
                             "text": "TEST - {CONTEXT} and {CODE}"}]
                            }
                         ]
                    }
                }
            }
        });
        Config::new(args).unwrap();
    }

    #[test]
    fn anthropic_config() {
        let args = json!({
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "model1": {
                        "type": "anthropic",
                        "completions_endpoint": "https://api.anthropic.com/v1/messages",
                        "model": "claude-3-haiku-20240307",
                        "auth_token_env_var_name": "ANTHROPIC_API_KEY",
                    },
                },
                "completion": {
                    "model": "model1",
                    "parameters": {
                        "system": "Test",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Test {CONTEXT} - {CODE}"
                            }
                        ],
                        "max_new_tokens": 32,
                    }
                }
            }
        });
        Config::new(args).unwrap();
    }
}

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
crates/lsp-ai/src/config.rs

# VSCode Open Tabs
crates/lsp-ai/src/transformer_backends/mod.rs
crates/lsp-ai/src/transformer_backends/ollama.rs
crates/lsp-ai/src/transformer_backends/open_ai/mod.rs
crates/lsp-ai/src/transformer_backends/gemini.rs
crates/lsp-ai/src/transformer_backends/anthropic.rs
crates/lsp-ai/src/transformer_backends/mistral_fim.rs
crates/lsp-ai/src/transformer_backends/llama_cpp/mod.rs
crates/lsp-ai/src/config.rs
crates/lsp-ai/src/transformer_worker.rs

# Current Time
4/24/2025, 2:30:20 AM (Australia/Sydney, UTC+10:00)

# Context Window Usage
133,713 / 1,048.576K tokens used (13%)

# Current Mode
PLAN MODE
In this mode you should focus on information gathering, asking questions, and architecting a solution. Once you have a plan, use the plan_mode_respond tool to engage in a conversational back and forth with the user. Do not use the plan_mode_respond tool until you've gathered all the information you need e.g. with read_file or ask_followup_question.
(Remember: If it seems the user wants you to use tools only available in Act Mode, you should ask the user to "toggle to Act mode" (use those words) - they will have to manually do this themselves with the Plan/Act toggle button below. You do not have the ability to switch to Act Mode yourself, and must wait for the user to do it themselves once they are satisfied with the plan. You also cannot present an option to toggle to Act mode, as this will be something you need to direct the user to do manually themselves.)
</environment_details>
