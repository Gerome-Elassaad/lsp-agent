use anyhow::Result;
use clap::Parser;
use directories::BaseDirs;
use lsp_server::{Connection, ExtractError, Message, Notification, Request, RequestId};
use lsp_types::{
    request::{CodeActionRequest, CodeActionResolveRequest, Completion, Shutdown},
    CodeActionOptions, CompletionOptions, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    RenameFilesParams, ServerCapabilities, TextDocumentSyncKind,
};
use std::sync::Mutex;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

mod config;
mod crawl;
mod custom_requests;
mod embedding_models;
mod memory_backends;
mod memory_worker;
mod project_intelligence;
mod splitters;
#[cfg(feature = "llama_cpp")]
mod template;
mod transformer_backends;
mod transformer_worker;
mod utils;

use config::Config;
use custom_requests::generation::Generation;
use memory_backends::MemoryBackend;
use transformer_backends::TransformerBackend;
use transformer_worker::{CompletionRequest, GenerationRequest, WorkerRequest};

use crate::{
    custom_requests::generation_stream::GenerationStream,
    transformer_worker::GenerationStreamRequest,
};

fn notification_is<N: lsp_types::notification::Notification>(notification: &Notification) -> bool {
    notification.method == N::METHOD
}

fn request_is<R: lsp_types::request::Request>(request: &Request) -> bool {
    request.method == R::METHOD
}

fn cast<R>(req: Request) -> Result<(RequestId, R::Params), ExtractError<Request>>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
}

// lsp-ai parameters
#[derive(Parser)]
#[command(version)]
struct Args {
    // Whether to use a custom log file
    #[arg(long, default_value_t = false)]
    use_seperate_log_file: bool,
    // A dummy argument for now
    #[arg(long, default_value_t = true)]
    stdio: bool,
    // JSON configuration file location
    #[arg(long, value_parser = utils::validate_file_exists, required = false)]
    config: Option<PathBuf>,
}

fn create_log_file(base_path: &Path) -> anyhow::Result<fs::File> {
    let dir_path = base_path.join("lsp-ai");
    fs::create_dir_all(&dir_path)?;
    let file_path = dir_path.join("lsp-ai.log");
    Ok(fs::File::create(file_path)?)
}

// Builds a tracing subscriber from the `LSP_AI_LOG` environment variable
// If the variables value is malformed or missing, sets the default log level to ERROR
fn init_logger(args: &Args) {
    let builder = FmtSubscriber::builder().with_env_filter(EnvFilter::from_env("LSP_AI_LOG"));
    let base_dirs = BaseDirs::new();

    if args.use_seperate_log_file && base_dirs.is_some() {
        let base_dirs = base_dirs.unwrap();
        let cache_dir = base_dirs.cache_dir();
        // Linux:   /home/alice/.cache
        // Windows: C:\Users\Alice\AppData\Local
        // macOS:   /Users/Alice/Library/Caches
        match create_log_file(&cache_dir) {
            Ok(log_file) => builder.with_writer(Mutex::new(log_file)).init(),
            Err(e) => {
                eprintln!("creating log file: {e:?} - falling back to stderr");
                builder
                    .with_writer(std::io::stderr)
                    .without_time()
                    .with_ansi(false)
                    .init()
            }
        }
    } else {
        builder
            .with_writer(std::io::stderr)
            .without_time()
            .with_ansi(false)
            .init()
    }
}

fn load_config(args: &Args, init_args: serde_json::Value) -> anyhow::Result<serde_json::Value> {
    if let Some(config_path) = &args.config {
        let config_data = fs::read_to_string(config_path)?;
        let mut config = serde_json::from_str(&config_data)?;
        utils::merge_json(&mut config, &init_args);
        Ok(config)
    } else {
        Ok(init_args)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_logger(&args);
    info!("lsp-ai logger initialized starting server");

    let (connection, io_threads) = Connection::stdio();
    let server_capabilities = serde_json::to_value(ServerCapabilities {
        completion_provider: Some(CompletionOptions::default()),
        text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::INCREMENTAL,
        )),
        code_action_provider: Some(lsp_types::CodeActionProviderCapability::Options(
            CodeActionOptions {
                resolve_provider: Some(true),
                ..Default::default()
            },
        )),
        ..Default::default()
    })?;
    let initialization_args = connection.initialize(server_capabilities)?;

    if let Err(e) = main_loop(connection, load_config(&args, initialization_args)?) {
        error!("{e:?}");
    }

    io_threads.join()?;
    Ok(())
}
</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
crates/lsp-ai/src/main.rs

# Open Tabs
crates/lsp-ai/src/project_intelligence/codebase_analysis/mod.rs
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Current Time
4/24/2025, 3:03:54 AM (Australia/Sydney, UTC+10:00)

# Context Window Usage
335,108 / 1,048.576K tokens used (32%)

# Current Mode
ACT MODE
</environment_details>

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
crates/lsp-ai/src/main.rs

# Open Tabs
crates/lsp-ai/src/project_intelligence/codebase_analysis/mod.rs
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Current Time
4/24/2025, 3:03:57 AM (Australia/Sydney, UTC+10:00)

# Context Window Usage
349,626 / 1,048.576K tokens used (33%)

# Current Mode
ACT MODE
</environment_details>
