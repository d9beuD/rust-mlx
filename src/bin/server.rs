//! Local OpenAI-style text API. MLX stays entirely in one owning OS thread.
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use rust_mlx::{
    chat::ChatTemplate,
    hybrid::HybridModel,
    mtp::Mtp,
    speculative::{self, Options},
    weights::Weights,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 3)]
    draft_depth: usize,
    #[arg(long)]
    no_mtp: bool,
    #[arg(long, default_value_t = 4096)]
    max_context: usize,
    #[arg(long, default_value_t = 8)]
    queue_capacity: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: Option<String>,
    prompt: Option<String>,
    messages: Option<Vec<Value>>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    #[serde(default = "thinking")]
    enable_thinking: bool,
    #[serde(default = "effort")]
    reasoning_effort: String,
    mtp: Option<bool>,
}
fn thinking() -> bool {
    true
}
fn effort() -> String {
    "xhigh".into()
}
struct Job {
    request: Request,
    chat: bool,
    id: String,
    events: mpsc::Sender<String>,
}
#[derive(Clone)]
struct App {
    jobs: mpsc::Sender<Job>,
    model: String,
    counter: Arc<AtomicU64>,
}
fn failure(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"invalid_request_error"}})),
    )
        .into_response()
}
async fn health() -> Json<Value> {
    Json(json!({"status":"ready","scheduler":"one MLX worker; bounded FIFO","batch_size":1}))
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":app.model,"object":"model","owned_by":"local"}]}))
}
async fn completion(State(app): State<App>, Json(request): Json<Request>) -> Response {
    dispatch(app, request, false).await
}
async fn chat(State(app): State<App>, Json(request): Json<Request>) -> Response {
    dispatch(app, request, true).await
}
async fn dispatch(app: App, request: Request, chat: bool) -> Response {
    if request.model.as_ref().is_some_and(|m| m != &app.model) {
        return failure(StatusCode::NOT_FOUND, "model not loaded");
    }
    if request.temperature.unwrap_or(0.) != 0.
        || request.top_p.unwrap_or(1.) != 1.
        || request.n.unwrap_or(1) != 1
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "this verified runtime supports temperature=0, top_p=1 and n=1",
        );
    }
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return failure(StatusCode::BAD_REQUEST, "choose one token limit");
    }
    let limit = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(256);
    if !(1..=4096).contains(&limit)
        || !["low", "medium", "xhigh"].contains(&request.reasoning_effort.as_str())
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid token limit or reasoning_effort",
        );
    }
    if chat && (request.messages.as_ref().is_none_or(Vec::is_empty) || request.prompt.is_some()) {
        return failure(
            StatusCode::BAD_REQUEST,
            "chat requires nonempty messages and no prompt",
        );
    }
    if !chat && (request.prompt.as_ref().is_none_or(String::is_empty) || request.messages.is_some())
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "completion requires a nonempty prompt and no messages",
        );
    }
    if chat
        && request.messages.as_ref().unwrap().iter().any(|m| {
            !m["content"].is_string()
                || !matches!(m["role"].as_str(), Some("user" | "assistant" | "system"))
                || m.get("tool_calls").is_some()
        })
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "only text messages without tools are supported",
        );
    }
    let stream = request.stream;
    let (tx, mut rx) = mpsc::channel(64);
    let id = format!("cmpl-rust-{}", app.counter.fetch_add(1, Ordering::Relaxed));
    match app.jobs.try_send(Job {
        request,
        chat,
        id,
        events: tx,
    }) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            return failure(StatusCode::TOO_MANY_REQUESTS, "inference queue full");
        }
        Err(_) => return failure(StatusCode::SERVICE_UNAVAILABLE, "MLX worker unavailable"),
    }
    if stream {
        let events =
            ReceiverStream::new(rx).map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match rx.recv().await {
            Some(data) => match serde_json::from_str::<Value>(&data) {
                Ok(v) if v.get("error").is_some() => {
                    (StatusCode::BAD_REQUEST, Json(v)).into_response()
                }
                Ok(v) => Json(v).into_response(),
                Err(_) => failure(StatusCode::INTERNAL_SERVER_ERROR, "invalid worker response"),
            },
            None => failure(StatusCode::SERVICE_UNAVAILABLE, "worker disconnected"),
        }
    }
}
fn run_job(
    job: &Job,
    model: &HybridModel,
    draft: Option<&Mtp>,
    tokenizer: &tokenizers::Tokenizer,
    template: &ChatTemplate,
    args: &Args,
    model_id: &str,
) -> Result<()> {
    ensure!(!job.events.is_closed(), "client disconnected");
    let r = &job.request;
    let prompt = if job.chat {
        template.render(
            r.messages.as_ref().unwrap(),
            r.enable_thinking,
            &r.reasoning_effort,
        )?
    } else {
        r.prompt.as_ref().unwrap().clone()
    };
    let ids = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let max = r.max_tokens.or(r.max_completion_tokens).unwrap_or(256);
    ensure!(
        !ids.is_empty() && ids.len() + max <= args.max_context,
        "prompt plus completion exceeds max_context ({})",
        args.max_context
    );
    let enabled = r.mtp.unwrap_or(!args.no_mtp);
    ensure!(!enabled || draft.is_some(), "MTP was disabled at startup");
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let object = if job.chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let base = |choice: Value| json!({"id":job.id,"object":object,"created":created,"model":model_id,"choices":[choice]});
    if r.stream && job.chat {
        job.events.blocking_send(
            base(json!({"index":0,"delta":{"role":"assistant"},"finish_reason":null})).to_string(),
        )?;
    }
    let mut decoder = tokenizer.decode_stream(true);
    let mut output = String::new();
    let emit = |token| -> Result<()> {
        ensure!(!job.events.is_closed(), "client disconnected");
        if let Some(text) = decoder.step(token).map_err(|e| anyhow::anyhow!("{e}"))? {
            output.push_str(&text);
            if r.stream {
                let choice = if job.chat {
                    json!({"index":0,"delta":{"content":text},"finish_reason":null})
                } else {
                    json!({"index":0,"text":text,"finish_reason":null})
                };
                job.events.blocking_send(base(choice).to_string())?;
            }
        }
        Ok(())
    };
    let options = Options {
        max_tokens: max,
        depth: args.draft_depth,
        chunk: 128,
        eos: &[248044, 248046],
    };
    let generation = if enabled {
        speculative::generate(model, draft.unwrap(), &ids, &options, emit)?
    } else {
        speculative::generate_plain(model, &ids, &options, emit)?
    };
    let reason = if generation.tokens.len() == max {
        "length"
    } else {
        "stop"
    };
    if r.stream {
        let choice = if job.chat {
            json!({"index":0,"delta":{},"finish_reason":reason})
        } else {
            json!({"index":0,"text":"","finish_reason":reason})
        };
        job.events.blocking_send(base(choice).to_string())?;
        job.events.blocking_send("[DONE]".into())?;
    } else {
        // Full decode also preserves a final incomplete byte sequence as the tokenizer specifies.
        output = tokenizer
            .decode(&generation.tokens, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let choice = if job.chat {
            json!({"index":0,"message":{"role":"assistant","content":output},"finish_reason":reason})
        } else {
            json!({"index":0,"text":output,"finish_reason":reason})
        };
        let mut response = base(choice);
        response["object"] = Value::String(
            if job.chat {
                "chat.completion"
            } else {
                "text_completion"
            }
            .into(),
        );
        response["usage"] = json!({"prompt_tokens":ids.len(),"completion_tokens":generation.tokens.len(),"total_tokens":ids.len()+generation.tokens.len()});
        response["rust_mlx"] = serde_json::to_value(&generation)?;
        job.events.blocking_send(response.to_string())?;
    }
    eprintln!(
        "{} prompt={} output={} MTP={} prefill={:.3}s decode={:.3}s",
        job.id,
        ids.len(),
        generation.tokens.len(),
        enabled,
        generation.prefill_seconds,
        generation.decode_seconds
    );
    Ok(())
}
fn worker(
    args: Args,
    mut jobs: mpsc::Receiver<Job>,
    ready: oneshot::Sender<std::result::Result<(), String>>,
    model_id: String,
) -> Result<()> {
    let loaded = (|| -> Result<_> {
        mlx_rs::memory::set_memory_limit(100 * 1024usize.pow(3))?;
        mlx_rs::memory::set_cache_limit(512 * 1024usize.pow(2))?;
        let weights = Weights::load(&args.model)?;
        let model = HybridModel::load(&weights, &args.model)?;
        let draft = if args.no_mtp {
            None
        } else {
            Some(Mtp::load(&weights, &model.config)?)
        };
        let tokenizer = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let template = ChatTemplate::load(&args.model)?;
        mlx_rs::transforms::eval(weights.tensors.values())?;
        Ok((weights, model, draft, tokenizer, template))
    })();
    let (_weights, model, draft, tokenizer, template) = match loaded {
        Ok(v) => {
            let _ = ready.send(Ok(()));
            v
        }
        Err(e) => {
            let _ = ready.send(Err(format!("{e:#}")));
            return Err(e);
        }
    };
    while let Some(job) = jobs.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        if let Err(e) = run_job(
            &job,
            &model,
            draft.as_ref(),
            &tokenizer,
            &template,
            &args,
            &model_id,
        ) {
            if !job.events.is_closed() {
                let _ = job.events.blocking_send(
                    json!({"error":{"message":format!("{e:#}"),"type":"inference_error"}})
                        .to_string(),
                );
                if job.request.stream {
                    let _ = job.events.blocking_send("[DONE]".into());
                }
            }
            eprintln!("{} failed: {e:#}", job.id);
        }
    }
    Ok(())
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=7).contains(&args.draft_depth) && args.queue_capacity > 0 && args.max_context > 0,
        "invalid server limits"
    );
    let listen = args.listen;
    let model_id = args
        .model
        .file_name()
        .context("model directory")?
        .to_string_lossy()
        .into_owned();
    let (tx, rx) = mpsc::channel(args.queue_capacity);
    let (ready_tx, ready_rx) = oneshot::channel();
    let worker_id = model_id.clone();
    let handle = std::thread::Builder::new()
        .name("mlx-inference".into())
        .spawn(move || worker(args, rx, ready_tx, worker_id))?;
    ready_rx
        .await
        .context("MLX worker failed during startup")?
        .map_err(anyhow::Error::msg)?;
    let app = App {
        jobs: tx,
        model: model_id,
        counter: Arc::new(AtomicU64::new(1)),
    };
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    eprintln!("ready: http://{listen}");
    let served = axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("MLX worker panicked"))??;
    served?;
    Ok(())
}
