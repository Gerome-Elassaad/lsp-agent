mod config;
mod custom_requests;
mod embedding_models;
mod memory_backends;
mod project_intelligence;
mod splitters;
mod transformer_backends;
mod transformer_worker;
mod utils;

use lsp_server::{Connection, ExtractError, Notification, Request, RequestId, Response};
use lsp_types::{
    CodeAction, CodeActionParams, CompletionItem, CompletionParams, CompletionResponse,
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, InitializeParams, InitializeResult,
    InitializedParams, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    error::Error,
    sync::{mpsc, Arc},
    thread,
};
use tracing::{error, info, instrument};

use crate::{
    config::Config,
    custom_requests::{generation::GENERATION_REQUEST, generation_stream::GENERATION_STREAM_REQUEST},
    memory_backends::MemoryBackend,
    transformer_backends::TransformerBackend,
    transformer_worker::{WorkerRequest, CompletionRequest, GenerationRequest, GenerationStreamRequest, CodeActionRequest, CodeActionResolveRequest},
};

#[instrument]
pub fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Tracing to file
    // let subscriber = tracing_subscriber::FmtSubscriber::builder()
    //     .with_max_level(tracing::Level::TRACE)
    //     .with_writer(std::io::stderr)
    //     .finish();
    // tracing::subscriber::set_global_default(subscriber)?;

    info!("starting lsp server");

    // LSP Setup (see https://github.com/rust-analyzer/rust-analyzer/blob/master/crates/lsp-server/src/lib.rs)
    let (connection, io_threads) = Connection::stdio();

    // Run the server and wait for the main thread to exit (shutdown).
    let server_capabilities = serde_json::to_value(ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::FULL,
        )),
        completion_provider: Some(Default::default()),
        code_action_provider: Some(lsp_types::CodeActionProviderCapability::Simple(true)),
        ..ServerCapabilities::default()
    })?;
    let initialization_params = connection.initialize(server_capabilities)?;
    main_loop(connection, initialization_params)?;
    io_threads.join()?;

    info!("shutting down lsp server");
    Ok(())
}

#[instrument]
fn main_loop(
    connection: Connection,
    initialization_params: InitializeParams,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("starting main loop");

    let init_params = initialization_params.process_id.map(|id| {
        info!("parent process id: {}", id)
    });

    let initialize_params = initialization_params.initialization_options.map(|options| {
        info!("initialization options: {:?}", options);
        options
    }).unwrap_or_else(|| {
        info!("no initialization options provided");
        Value::Null
    });

    let config = Config::new(initialize_params)?;

    let (memory_tx, memory_rx) = mpsc::channel();
    let (transformer_tx, transformer_rx) = mpsc::channel();

    // Start Memory Backend
    let memory_backend: Box<dyn MemoryBackend + Send + Sync> = match &config.config.memory {
        config::ValidMemoryBackend::FileStore(file_store) => Box::new(
            memory_backends::file_store::FileStore::new(
                config.client_params.root_uri.clone(),
                file_store.clone(),
            )?,
        ),
        config::ValidMemoryBackend::VectorStore(vector_store) => Box::new(
            memory_backends::vector_store::VectorStore::new(
                config.client_params.root_uri.clone(),
                vector_store.clone(),
            )?,
        ),
        config::ValidMemoryBackend::PostgresML(postgresml) => Box::new(
            memory_backends::postgresml::PostgresML::new(
                config.client_params.root_uri.clone(),
                postgresml.clone(),
            )?,
        ),
    };
    thread::spawn(move || memory_worker::run(memory_backend, memory_rx));

    // Start Transformer Backend
    let mut transformer_backends: HashMap<String, Box<dyn TransformerBackend + Send + Sync>> =
        HashMap::new();
    for (model_key, model) in config.config.models.iter() {
        transformer_backends.insert(model_key.clone(), model.clone().try_into()?);
    }
    let transformer_backends = Arc::new(transformer_backends);
    let connection_arc = Arc::new(connection);
    let config_clone = config.clone();
    thread::spawn(move || {
        transformer_worker::run(
            transformer_backends,
            memory_tx,
            transformer_rx,
            connection_arc,
            config_clone,
        )
    });

    let capabilities = get_server_capabilities(&config);
    let initialize_result = InitializeResult {
        capabilities,
        server_info: None,
    };
    info!("Initialization complete, starting main loop");
    let codebase_analyzer = project_intelligence::codebase_analysis::CodebaseAnalyzer::new(config.clone());

    for msg in &connection.receiver {
        // TESTING - remove this later
        let code = r#"
            fn hello_world() {
                println!("Hello, world!");
            }
        "#;
        let language = "rust";
        let tree = codebase_analyzer.analyze_code(code, language);
        info!("Tree: {:?}", tree);

        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                info!("request: {:?}", req);
                let request = req.clone();
                let transformer_tx_clone = transformer_tx.clone();
                let connection_clone = connection.clone();
                tokio::spawn(async move {
                    handle_request(request, transformer_tx_clone, connection_clone).await;
                });
            }
            Message::Notification(notification) => {
                info!("notification: {:?}", notification);
                let notification_clone = notification.clone();
                let memory_tx_clone = memory_tx.clone();
                let params = cast_notification::<lsp_types::DidOpenTextDocumentParams>(notification).unwrap();
                let code = params.text_document.text.clone();
                let language = params.text_document.language_id.clone();
                let tree = codebase_analyzer.analyze_code(&code, &language);
                info!("Tree: {:?}", tree);
                tokio::spawn(async move {
                    handle_notification(notification_clone, memory_tx_clone, &codebase_analyzer).await;
                });
            }
            Message::Response(resp) => {
                info!("response: {:?}", resp)
            }
        }
    }

    Ok(())
}

#[instrument]
async fn handle_request(
    req: Request,
    transformer_tx: mpsc::Sender<WorkerRequest>,
    connection: Connection,
) {
    let request_id = req.id.clone();
    match req.method.as_str() {
        "textDocument/completion" => {
            let (id, params) = cast::<lsp_types::request::Completion>(req).unwrap();
            let completion_request = CompletionRequest::new(id, params);
            if let Err(e) = transformer_tx.send(WorkerRequest::Completion(completion_request)) {
                error!("error sending completion request: {e:?}");
            }
        }
        GENERATION_REQUEST => {
            let (id, params) = cast::<crate::custom_requests::generation::Generation>(req).unwrap();
            let generation_request = GenerationRequest::new(id, params);
            if let Err(e) = transformer_tx.send(WorkerRequest::Generation(generation_request)) {
                error!("error sending generation request: {e:?}");
            }
        }
        GENERATION_STREAM_REQUEST => {
            let (id, params) = cast::<crate::custom_requests::generation_stream::GenerationStream>(req).unwrap();
            let generation_stream_request = GenerationStreamRequest::new(id, params);
            if let Err(e) = transformer_tx.send(WorkerRequest::GenerationStream(generation_stream_request)) {
                error!("error sending generation stream request: {e:?}");
            }
        }
        "textDocument/codeAction" => {
            let (id, params) = cast::<lsp_types::request::CodeAction>(req).unwrap();
            let code_action_request = CodeActionRequest::new(id, params);
            if let Err(e) = transformer_tx.send(WorkerRequest::CodeActionRequest(code_action_request)) {
                error!("error sending code action request: {e:?}");
            }
        }
        "codeAction/resolve" => {
            let (id, params) = cast::<lsp_types::request::CodeActionResolve>(req).unwrap();
            let code_action_resolve_request = CodeActionResolveRequest::new(id, params);
            if let Err(e) = transformer_tx.send(WorkerRequest::CodeActionResolveRequest(code_action_resolve_request)) {
                error!("error sending code action resolve request: {e:?}");
            }
        }
        _ => {
            error!("unsupported request: {:?}", req.method);
            let resp = Response {
                id: request_id,
                result: None,
                error: Some(lsp_server::ResponseError::new_not_found()),
            };
            connection.sender.send(Message::Response(resp)).unwrap();
        }
    }
}

#[instrument]
async fn handle_notification(notification: Notification, memory_tx: mpsc::Sender<memory_worker::WorkerRequest>, codebase_analyzer: &project_intelligence::codebase_analysis::CodebaseAnalyzer) {
    info!("handle_notification: {:?}", notification.method);

    match notification.method.as_str() {
        "textDocument/didOpen" => {
            let params = cast_notification::<lsp_types::DidOpenTextDocumentParams>(notification).unwrap();
            let code = params.text_document.text.clone();
            let language = params.text_document.language_id.clone();
            let tree = codebase_analyzer.analyze_code(&code, &language);
            info!("Tree: {:?}", tree);

            if let Err(e) = memory_tx.send(memory_worker::WorkerRequest::AddFile(params.text_document)) {
                error!("error sending didOpen notification: {e:?}");
            }
        }
        "textDocument/didChange" => {
            let params = cast_notification::<lsp_types::DidChangeTextDocumentParams>(notification).unwrap();
            info!("textDocument/didChange notification received");
            let code = params.content_changes[0].text.clone();
            // TODO: Get language from text document
            let language = "rust";
            //let tree = codebase_analyzer.analyze_code(&code, language);
            //info!("Tree: {:?}", tree);

            if let Err(e) = memory_tx.send(memory_worker::WorkerRequest::UpdateFile(params)) {
                error!("error sending didChange notification: {e:?}");
            }
        }
        _ => {
            info!("unsupported notification: {:?}", notification.method);
        }
    }
}

fn get_server_capabilities(config: &Config) -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        completion_provider: Some(Default::default()),
        code_action_provider: Some(lsp_types::CodeActionProviderCapability::Simple(true)),
        ..ServerCapabilities::default()
    }
}

fn cast<R>(req: Request) -> Result<(RequestId, R::Params), ExtractError<Request>>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
}

fn cast_notification<N>(notification: Notification) -> Result<N::Params, ExtractError<Notification>>
where
    N: lsp_types::notification::Notification,
    N::Params: serde::de::DeserializeOwned,
{
    notification.extract(N::METHOD)
}

fn notification_is<N: lsp_types::notification::Notification>(notification: &Notification) -> bool {
    notification.method == N::METHOD
}

fn request_is<R: lsp_types::request::Request>(request: &Request) -> bool {
    request.method == R::METHOD
}
</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Open Tabs
crates/lsp-ai/src/project_intelligence/codebase_analysis/mod.rs
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Current Time
4/24/2025, 3:01:51 AM (Australia/Sydney, UTC+10:00)

# Context Window Usage
301,774 / 1,048.576K tokens used (29%)

# Current Mode
ACT MODE
</environment_details>

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Open Tabs
crates/lsp-ai/src/project_intelligence/codebase_analysis/mod.rs
crates/lsp-ai/src/lib.rs
crates/lsp-ai/src/main.rs

# Current Time
4/24/2025, 3:03:31 AM (Australia/Sydney, UTC+10:00)

# Context Window Usage
336,247 / 1,048.576K tokens used (32%)

# Current Mode
ACT MODE
</environment_details>
