//! JSON-RPC 2.0 stdio server for voice TTS.
//!
//! Reads newline-delimited JSON-RPC requests from stdin, writes responses to
//! stdout. The model and a default voice are loaded once at startup; callers
//! can switch voices mid-session.
//!
//! ## Methods
//!
//! - `speak`        — Generate and play audio from text or phonemes.
//! - `listen`       — Record from mic (VAD auto-stop), transcribe, return text.
//! - `cancel`       — Interrupt the current utterance's playback or recording.
//! - `set_voice`    — Change the session's default voice.
//! - `set_speed`    — Change the session's default speed.
//! - `list_voices`  — Return the list of builtin voice names.
//! - `ping`         — Health check, returns `"pong"`.
//!
//! ## Example session (stdin → stdout)
//!
//! ```jsonl
//! → {"jsonrpc":"2.0","method":"speak","params":{"text":"Hello world"},"id":1}
//! ← {"jsonrpc":"2.0","result":{"duration_ms":1840,"chunks":1},"id":1}
//!
//! → {"jsonrpc":"2.0","method":"speak","params":{"text":"Hello. How are you?","detail":"full"},"id":2}
//! ← {"jsonrpc":"2.0","method":"speak.progress","params":{"chunk":1,"total":2,"phonemes":"həlˈO."}}
//! ← {"jsonrpc":"2.0","method":"speak.progress","params":{"chunk":2,"total":2,"phonemes":"hˌaʊ ɑːɹ ɪU?"}}
//! ← {"jsonrpc":"2.0","result":{"duration_ms":2100,"chunks":2,"phonemes":["həlˈO.","hˌaʊ ɑːɹ ɪU?"]},"id":2}
//!
//! → {"jsonrpc":"2.0","method":"set_voice","params":{"voice":"am_michael"},"id":3}
//! ← {"jsonrpc":"2.0","result":{"voice":"am_michael"},"id":3}
//!
//! → {"jsonrpc":"2.0","method":"list_voices","id":4}
//! ← {"jsonrpc":"2.0","result":{"voices":["af_heart","af_bella","af_sarah","af_sky","am_michael","am_adam","bf_emma"]},"id":4}
//! ```
//!
//! Notifications (requests without `id`) are executed but produce no response.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::num::NonZero;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;

use crate::{
    apply_substitutions, apply_tech_subs, collect_subs, interrupted, listen, strip_markdown,
    INTERRUPTED, QUIET,
};

// ── JSON-RPC 2.0 types ────────────────────────────────────────────────

const JSONRPC_VERSION: &str = "2.0";

#[derive(Debug, Deserialize)]
struct Request {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    method: String,
    #[serde(default)]
    params: Value,
    id: Option<Value>,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
    id: Value,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// Standard JSON-RPC error codes
const PARSE_ERROR: i64 = -32700;
#[allow(dead_code)]
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

impl Response {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
            id,
        }
    }
}

// ── Method params ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SpeakParams {
    /// Text to speak (mutually exclusive with `phonemes`).
    text: Option<String>,
    /// Raw phoneme string, bypasses G2P.
    phonemes: Option<String>,
    /// Override voice for this utterance only.
    voice: Option<String>,
    /// Override speed for this utterance only.
    speed: Option<f32>,
    /// Strip markdown/MDX formatting before G2P conversion.
    #[serde(default)]
    markdown: bool,
    /// Response detail level: "normal" (default) or "full" (includes phonemes).
    #[serde(default)]
    detail: Detail,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Detail {
    #[default]
    Normal,
    Full,
}

#[derive(Debug, Deserialize)]
struct SetVoiceParams {
    voice: String,
}

#[derive(Debug, Deserialize)]
struct SetSpeedParams {
    speed: f32,
}

#[derive(Debug, Deserialize)]
struct ListenParams {
    /// Maximum recording duration in milliseconds (default: 30000).
    max_duration_ms: Option<u64>,
    /// Stop after this many ms of silence following speech (default: 2000).
    silence_timeout_ms: Option<u64>,
    /// Minimum amplitude threshold for voice activity detection (default: 0.01).
    /// The actual threshold is max(this, noise_floor × noise_multiplier).
    silence_threshold: Option<f32>,
    /// Multiplier applied to the calibrated noise floor to set the adaptive
    /// threshold (default: 3.0). Higher = less sensitive, lower = more sensitive.
    noise_multiplier: Option<f32>,
    /// Duration in milliseconds to calibrate the noise floor at the start
    /// of recording (default: 500). Set to 0 to skip calibration and use
    /// silence_threshold directly.
    calibration_ms: Option<u64>,
}

// ── Session state ──────────────────────────────────────────────────────

struct Session {
    model: voice_tts::KokoroModel,
    voice: voice_tts::Array,
    voice_name: String,
    speed: f32,
    sample_rate: u32,
    repo_id: String,
    subs: Vec<(String, String)>,
    phoneme_overrides: HashMap<String, String>,
    /// Cache of loaded voices so we don't re-load on every `speak`.
    voice_cache: HashMap<String, voice_tts::Array>,
    /// Lazily-loaded STT model (only initialized on first `listen` call).
    stt_model: Option<voice_stt::MoonshineModel>,
    /// Lazily-loaded STT tokenizer.
    stt_tokenizer: Option<voice_stt::tokenizers::Tokenizer>,
}

impl Session {
    fn get_voice(&mut self, name: &str) -> Result<&voice_tts::Array, String> {
        if !self.voice_cache.contains_key(name) {
            let v = voice_tts::load_voice(name, Some(&self.repo_id))
                .map_err(|e| format!("Failed to load voice '{name}': {e}"))?;
            self.voice_cache.insert(name.to_string(), v);
        }
        Ok(&self.voice_cache[name])
    }
}

// ── Public entry point ─────────────────────────────────────────────────

/// Configuration for starting the JSON-RPC server, bundled to avoid
/// passing too many positional arguments.
pub struct ServerConfig {
    pub model: voice_tts::KokoroModel,
    pub voice: voice_tts::Array,
    pub voice_name: String,
    pub speed: f32,
    pub sample_rate: u32,
    pub repo_id: String,
    pub cli_subs: Vec<String>,
    pub sub_file_path: Option<std::path::PathBuf>,
}

/// Parsed message from the stdin reader thread.
enum StdinMsg {
    Request(Request),
    ParseError(String),
    Closed,
}

/// Run the JSON-RPC stdio server. Blocks until stdin is closed or interrupted.
///
/// Stdin is read on a dedicated thread so that `cancel` requests can arrive
/// while a `speak` call is still generating/playing audio. The main loop
/// pulls parsed messages from an mpsc channel.
pub fn run(config: ServerConfig) {
    let (subs, phoneme_overrides) = collect_subs(&config.cli_subs, config.sub_file_path.as_deref());

    let mut voice_cache = HashMap::new();
    voice_cache.insert(config.voice_name.clone(), config.voice.clone());

    let mut session = Session {
        model: config.model,
        voice: config.voice,
        voice_name: config.voice_name,
        speed: config.speed,
        sample_rate: config.sample_rate,
        repo_id: config.repo_id,
        subs,
        phoneme_overrides,
        voice_cache,
        stt_model: None,
        stt_tokenizer: None,
    };

    let mut stdout = io::stdout();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("voice jsonrpc server ready");
    }

    // Spawn a reader thread so stdin isn't blocked while speak is running.
    // This lets `cancel` arrive mid-playback.
    let (tx, rx) = mpsc::channel::<StdinMsg>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let msg = match line {
                Ok(l) => {
                    let l = l.trim().to_string();
                    if l.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Request>(&l) {
                        Ok(req) => {
                            // Set the interrupt flag immediately on the reader
                            // thread so it takes effect while speak is blocked
                            // on the main thread.
                            if req.method == "cancel" {
                                INTERRUPTED.store(true, Ordering::SeqCst);
                            }
                            StdinMsg::Request(req)
                        }
                        Err(e) => StdinMsg::ParseError(format!("Parse error: {e}")),
                    }
                }
                Err(_) => StdinMsg::Closed,
            };
            let is_closed = matches!(msg, StdinMsg::Closed);
            if tx.send(msg).is_err() || is_closed {
                break;
            }
        }
        // EOF — signal the main loop
        let _ = tx.send(StdinMsg::Closed);
    });

    // Main dispatch loop — drains the channel, handles requests.
    // Between speak chunks we also drain any queued cancel requests
    // via the interrupted() flag (set by handle_cancel or Ctrl+C).
    while let Ok(msg) = rx.recv() {
        match msg {
            StdinMsg::Closed => break,
            StdinMsg::ParseError(e) => {
                let resp = Response::error(Value::Null, PARSE_ERROR, e);
                write_response(&mut stdout, &resp);
            }
            StdinMsg::Request(req) => {
                let is_notification = req.id.is_none();
                let id = req.id.clone().unwrap_or(Value::Null);

                let resp = dispatch(&mut session, &mut stdout, &req.method, req.params, id);

                if !is_notification {
                    if let Some(resp) = resp {
                        write_response(&mut stdout, &resp);
                    }
                }

                // Reset so the next speak isn't pre-cancelled.
                INTERRUPTED.store(false, Ordering::Relaxed);

                // Drain any requests that arrived while we were busy (e.g.
                // a cancel that came in right at the tail end of playback).
                drain_pending(&rx, &mut session, &mut stdout);
            }
        }
    }
}

/// Process any already-queued messages without blocking.
fn drain_pending(rx: &mpsc::Receiver<StdinMsg>, session: &mut Session, stdout: &mut io::Stdout) {
    loop {
        match rx.try_recv() {
            Ok(StdinMsg::Request(req)) => {
                let is_notification = req.id.is_none();
                let id = req.id.clone().unwrap_or(Value::Null);
                let resp = dispatch(session, stdout, &req.method, req.params, id);
                if !is_notification {
                    if let Some(resp) = resp {
                        write_response(stdout, &resp);
                    }
                }
                INTERRUPTED.store(false, Ordering::Relaxed);
            }
            Ok(StdinMsg::ParseError(e)) => {
                let resp = Response::error(Value::Null, PARSE_ERROR, e);
                write_response(stdout, &resp);
            }
            _ => break, // empty or closed
        }
    }
}

// ── Dispatch ───────────────────────────────────────────────────────────

fn dispatch(
    session: &mut Session,
    stdout: &mut io::Stdout,
    method: &str,
    params: Value,
    id: Value,
) -> Option<Response> {
    let result = match method {
        "speak" => handle_speak(session, stdout, params),
        "listen" => handle_listen(session, params),
        "cancel" => handle_cancel(),
        "set_voice" => handle_set_voice(session, params),
        "set_speed" => handle_set_speed(session, params),
        "list_voices" => handle_list_voices(),
        "ping" => Ok(serde_json::json!("pong")),
        _ => {
            return Some(Response::error(
                id,
                METHOD_NOT_FOUND,
                format!("Unknown method: {method}"),
            ));
        }
    };

    Some(match result {
        Ok(value) => Response::success(id, value),
        Err(e) => e.into_response(id),
    })
}

// ── Error helper ───────────────────────────────────────────────────────

struct RpcErr {
    code: i64,
    message: String,
}

impl RpcErr {
    fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: msg.into(),
        }
    }

    fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: INTERNAL_ERROR,
            message: msg.into(),
        }
    }

    fn into_response(self, id: Value) -> Response {
        Response::error(id, self.code, self.message)
    }
}

// ── Method handlers ────────────────────────────────────────────────────

fn handle_cancel() -> Result<Value, RpcErr> {
    INTERRUPTED.store(true, Ordering::SeqCst);
    Ok(serde_json::json!({"cancelled": true}))
}

fn handle_listen(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
    let p: ListenParams = if params.is_null() {
        ListenParams {
            max_duration_ms: None,
            silence_timeout_ms: None,
            silence_threshold: None,
            noise_multiplier: None,
            calibration_ms: None,
        }
    } else {
        serde_json::from_value(params)
            .map_err(|e| RpcErr::invalid_params(format!("bad listen params: {e}")))?
    };

    let max_duration = p.max_duration_ms.unwrap_or(30_000);
    let silence_timeout = p.silence_timeout_ms.unwrap_or(2_000);
    let threshold = p.silence_threshold.unwrap_or(0.01);
    let noise_multiplier = p.noise_multiplier.unwrap_or(3.0);
    let calibration_ms = p.calibration_ms.unwrap_or(500);

    // Lazily load STT model on first listen call
    if session.stt_model.is_none() {
        let repo = std::env::var("STT_MODEL")
            .unwrap_or_else(|_| "UsefulSensors/moonshine-base".to_string());

        if !QUIET.load(Ordering::Relaxed) {
            eprintln!("Loading STT model ({repo})...");
        }

        let model = voice_stt::load_model(&repo)
            .map_err(|e| RpcErr::internal(format!("Failed to load STT model: {e}")))?;
        let tokenizer = voice_stt::load_tokenizer(&repo)
            .map_err(|e| RpcErr::internal(format!("Failed to load tokenizer: {e}")))?;

        session.stt_model = Some(model);
        session.stt_tokenizer = Some(tokenizer);

        if !QUIET.load(Ordering::Relaxed) {
            eprintln!("STT model loaded.");
        }
    }

    let stt_model = session.stt_model.as_mut().unwrap();
    let stt_tokenizer = session.stt_tokenizer.as_ref().unwrap();

    let started = Instant::now();

    let result = listen::listen_and_transcribe_vad(
        stt_model,
        stt_tokenizer,
        max_duration,
        silence_timeout,
        threshold,
        noise_multiplier,
        calibration_ms,
    );

    let duration_ms = started.elapsed().as_millis() as u64;

    // Reset interrupt flag after listen completes
    INTERRUPTED.store(false, Ordering::Relaxed);

    match result {
        Some(r) => Ok(serde_json::json!({
            "text": r.text,
            "tokens": r.tokens.len(),
            "duration_ms": duration_ms,
        })),
        None => Ok(serde_json::json!({
            "text": "",
            "tokens": 0,
            "duration_ms": duration_ms,
        })),
    }
}

fn handle_speak(
    session: &mut Session,
    stdout: &mut io::Stdout,
    params: Value,
) -> Result<Value, RpcErr> {
    let p: SpeakParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    let speed = p.speed.unwrap_or(session.speed);

    // Determine which voice to use
    let voice_ref: *const voice_tts::Array = if let Some(ref name) = p.voice {
        session.get_voice(name).map_err(RpcErr::invalid_params)? as *const _
    } else {
        &session.voice as *const _
    };
    // SAFETY: voice_ref points into session.voice or session.voice_cache,
    // both of which live for the duration of this call. We use a raw pointer
    // to avoid borrow-checker conflicts with the mutable session borrow for
    // the model below.
    let voice = unsafe { &*voice_ref };

    // Resolve phoneme chunks
    let chunks: Vec<String> = if let Some(ref phonemes) = p.phonemes {
        vec![phonemes.clone()]
    } else if let Some(ref text) = p.text {
        let text = if p.markdown {
            strip_markdown(text)
        } else {
            text.clone()
        };
        let text = apply_tech_subs(&text);
        let text = if session.subs.is_empty() {
            text
        } else {
            apply_substitutions(&text, &session.subs)
        };
        let result = if session.phoneme_overrides.is_empty() {
            voice_g2p::text_to_phoneme_chunks(&text)
        } else {
            voice_g2p::text_to_phoneme_chunks_with_overrides(&text, &session.phoneme_overrides)
        };
        result.map_err(|e| RpcErr::internal(format!("G2P error: {e}")))?
    } else {
        return Err(RpcErr::invalid_params(
            "Either 'text' or 'phonemes' is required",
        ));
    };

    // Generate and play
    let detail_full = p.detail == Detail::Full;
    let started = Instant::now();
    stream_chunks(session, stdout, voice, &chunks, speed, detail_full)?;
    let duration_ms = started.elapsed().as_millis() as u64;

    let mut result = serde_json::json!({
        "duration_ms": duration_ms,
        "chunks": chunks.len(),
    });

    if p.detail == Detail::Full {
        result["phonemes"] = serde_json::json!(chunks);
    }

    Ok(result)
}

fn handle_set_voice(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
    let p: SetVoiceParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    // Pre-load to validate the voice exists
    let voice = session
        .get_voice(&p.voice)
        .map_err(RpcErr::invalid_params)?
        .clone();

    session.voice = voice;
    session.voice_name = p.voice.clone();

    Ok(serde_json::json!({ "voice": p.voice }))
}

fn handle_set_speed(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
    let p: SetSpeedParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    if p.speed <= 0.0 || p.speed > 5.0 {
        return Err(RpcErr::invalid_params(
            "Speed must be between 0.0 (exclusive) and 5.0 (inclusive)",
        ));
    }

    session.speed = p.speed;

    Ok(serde_json::json!({ "speed": p.speed }))
}

fn handle_list_voices() -> Result<Value, RpcErr> {
    Ok(serde_json::json!({
        "voices": voice_tts::builtin::BUILTIN_VOICES,
    }))
}

// ── Audio playback ─────────────────────────────────────────────────────

fn stream_chunks(
    session: &mut Session,
    stdout: &mut io::Stdout,
    voice: &voice_tts::Array,
    chunks: &[String],
    speed: f32,
    progress: bool,
) -> Result<(), RpcErr> {
    use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};

    let mut stream =
        DeviceSinkBuilder::open_default_sink().map_err(|e| RpcErr::internal(e.to_string()))?;
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let channels = NonZero::new(1u16).unwrap();
    let rate = NonZero::new(session.sample_rate).unwrap();

    for (i, phonemes) in chunks.iter().enumerate() {
        if interrupted() {
            break;
        }
        if phonemes.is_empty() {
            continue;
        }
        if chunks.len() > 1 && !QUIET.load(Ordering::Relaxed) {
            eprintln!("  generating chunk {}/{}...", i + 1, chunks.len());
        }
        if progress {
            write_notification(
                stdout,
                "speak.progress",
                serde_json::json!({
                    "chunk": i + 1,
                    "total": chunks.len(),
                    "phonemes": phonemes,
                }),
            );
        }
        match voice_tts::generate(&mut session.model, phonemes, voice, speed) {
            Ok(audio) => {
                let samples: Vec<f32> = audio.as_slice().to_vec();
                let source = SamplesBuffer::new(channels, rate, samples);
                player.append(source);
            }
            Err(e) => {
                return Err(RpcErr::internal(format!(
                    "Generation failed on chunk {}: {e}",
                    i + 1
                )));
            }
        }
    }

    // Wait for playback, checking for Ctrl+C
    while !player.empty() {
        if interrupted() {
            player.stop();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

// ── IO ─────────────────────────────────────────────────────────────────

fn write_response(stdout: &mut io::Stdout, resp: &Response) {
    // Unwrap is fine — our Response type is always serializable.
    let json = serde_json::to_string(resp).unwrap();
    let _ = writeln!(stdout, "{json}");
    let _ = stdout.flush();
}

/// Emit a server-initiated notification (no `id`).
fn write_notification(stdout: &mut io::Stdout, method: &str, params: Value) {
    let msg = serde_json::json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params,
    });
    let json = serde_json::to_string(&msg).unwrap();
    let _ = writeln!(stdout, "{json}");
    let _ = stdout.flush();
}
