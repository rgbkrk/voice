//! MCP (Model Context Protocol) stdio server for voice TTS/STT.
//!
//! Implements the MCP protocol on top of JSON-RPC 2.0, exposing voice tools
//! (speak, converse, set_voice, set_speed, list_voices, set_start_sound, set_stop_sound, play_sound, cancel) to
//! MCP-compatible clients like Claude Code.
//!
//! ## Usage
//!
//!     voice mcp
//!
//! The server communicates over stdin/stdout using newline-delimited JSON-RPC.

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

const PARSE_ERROR: i64 = -32700;
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

// ── Voice method params ───────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize)]
struct SpeakParams {
    text: Option<String>,
    phonemes: Option<String>,
    voice: Option<String>,
    speed: Option<f32>,
    #[serde(default)]
    markdown: bool,
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
struct SetSoundParams {
    /// Path to a WAV file, or null/absent to reset to default.
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ListenParams {
    max_duration_ms: Option<u64>,
    silence_timeout_ms: Option<u64>,
    silence_threshold: Option<f32>,
    noise_multiplier: Option<f32>,
    calibration_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ConverseParams {
    text: Option<String>,
    phonemes: Option<String>,
    voice: Option<String>,
    speed: Option<f32>,
    #[serde(default)]
    markdown: bool,
    max_duration_ms: Option<u64>,
    silence_timeout_ms: Option<u64>,
    silence_threshold: Option<f32>,
    noise_multiplier: Option<f32>,
    calibration_ms: Option<u64>,
}

// ── Session state ─────────────────────────────────────────────────────

struct Session {
    model: voice_tts::KokoroModel,
    voice: voice_tts::Array,
    voice_name: String,
    speed: f32,
    sample_rate: u32,
    repo_id: String,
    subs: Vec<(String, String)>,
    phoneme_overrides: HashMap<String, String>,
    voice_cache: HashMap<String, voice_tts::Array>,
    stt_model: Option<voice_stt::MoonshineModel>,
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

// ── Public entry point ────────────────────────────────────────────────

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

enum StdinMsg {
    Request(Request),
    ParseError(String),
    Closed,
}

/// Run the MCP stdio server. Blocks until stdin is closed or interrupted.
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

    // Cap the Metal buffer cache at 2 GB to prevent unbounded memory growth
    // across repeated inference calls.
    if let Err(e) = quill_mlx::metal::set_cache_limit(2 * 1024 * 1024 * 1024) {
        eprintln!("warning: failed to set Metal cache limit: {e}");
    }

    let mut stdout = io::stdout();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("voice mcp server ready");
    }

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
        let _ = tx.send(StdinMsg::Closed);
    });

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

                INTERRUPTED.store(false, Ordering::Relaxed);
                drain_pending(&rx, &mut session, &mut stdout);
            }
        }
    }
}

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
            _ => break,
        }
    }
}

// ── MCP Protocol ──────────────────────────────────────────────────────

const SERVER_NAME: &str = "voice";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

fn dispatch(
    session: &mut Session,
    stdout: &mut io::Stdout,
    method: &str,
    params: Value,
    id: Value,
) -> Option<Response> {
    let result = match method {
        // MCP lifecycle
        "initialize" => handle_initialize(params),
        "notifications/initialized" => return None,

        // MCP tool discovery
        "tools/list" => handle_tools_list(),

        // MCP tool execution
        "tools/call" => {
            return Some(handle_tools_call(session, stdout, params, id));
        }

        // MCP ping
        "ping" => Ok(serde_json::json!({})),

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

fn handle_initialize(_params: Value) -> Result<Value, RpcErr> {
    Ok(serde_json::json!({
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    }))
}

fn handle_tools_list() -> Result<Value, RpcErr> {
    Ok(serde_json::json!({
        "tools": [
            {
                "name": "speak",
                "description": "Speak text aloud using text-to-speech. Plays audio through the default output device.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "Text to speak" },
                        "voice": { "type": "string", "description": "Voice name override for this utterance" },
                        "speed": { "type": "number", "description": "Speed override for this utterance" },
                        "markdown": { "type": "boolean", "description": "Strip markdown formatting before speaking" }
                    },
                    "required": ["text"]
                }
            },
            // listen is available internally (used by converse) but not exposed
            // as a standalone MCP tool — converse is strictly better for
            // interactive use since it combines speak+listen in one round trip.
            //
            // {
            //     "name": "listen",
            //     "description": "Record from microphone (plays a ding when ready), transcribe speech to text using VAD auto-stop. Blocks until speech is detected and silence timeout fires.",
            //     "inputSchema": {
            //         "type": "object",
            //         "properties": {
            //             "max_duration_ms": { "type": "number", "description": "Maximum recording duration in milliseconds (default: 30000)" },
            //             "silence_timeout_ms": { "type": "number", "description": "Stop after this many ms of silence following speech (default: 2000)" },
            //             "silence_threshold": { "type": "number", "description": "Minimum amplitude for voice activity detection (default: 0.01)" },
            //             "noise_multiplier": { "type": "number", "description": "Multiplier for calibrated noise floor (default: 3.0)" },
            //             "calibration_ms": { "type": "number", "description": "Duration in ms to calibrate noise floor (default: 500)" }
            //         }
            //     }
            // },
            {
                "name": "converse",
                "description": "Speak text aloud, then immediately listen for a response. Combines speak and listen into a single turn-based exchange, reducing round trips.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "Text to speak before listening" },
                        "voice": { "type": "string", "description": "Voice name override for this utterance" },
                        "speed": { "type": "number", "description": "Speed override for this utterance" },
                        "markdown": { "type": "boolean", "description": "Strip markdown formatting before speaking" },
                        "max_duration_ms": { "type": "number", "description": "Maximum listen duration in milliseconds (default: 30000)" },
                        "silence_timeout_ms": { "type": "number", "description": "Stop listening after this many ms of silence following speech (default: 2000)" },
                        "silence_threshold": { "type": "number", "description": "Minimum amplitude for voice activity detection (default: 0.01)" },
                        "noise_multiplier": { "type": "number", "description": "Multiplier for calibrated noise floor (default: 3.0)" },
                        "calibration_ms": { "type": "number", "description": "Duration in ms to calibrate noise floor (default: 500)" }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "cancel",
                "description": "Cancel the current speak or listen operation.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "set_voice",
                "description": "Change the default voice for subsequent speak calls.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "voice": { "type": "string", "description": "Voice name (e.g. af_heart, am_michael, am_adam)" }
                    },
                    "required": ["voice"]
                }
            },
            {
                "name": "set_speed",
                "description": "Change the default speech speed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "speed": { "type": "number", "description": "Speed factor (0.0 exclusive to 5.0 inclusive, 1.0 = normal)" }
                    },
                    "required": ["speed"]
                }
            },
            {
                "name": "list_voices",
                "description": "List available voices with metadata (id, name, language, gender, grade). By default returns only built-in voices for speed. Set 'all' to true to get the full catalog of 54 voices across 9 languages, including availability status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "all": {
                            "type": "boolean",
                            "description": "If true, return all 54 known voices with availability status (builtin, cached, or available for download). Default: false (builtin only)."
                        }
                    }
                }
            },
            {
                "name": "set_start_sound",
                "description": "Set a custom WAV file to play when listening starts (replaces the default ding). Omit path to reset to default.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to a WAV file" }
                    }
                }
            },
            {
                "name": "set_stop_sound",
                "description": "Set a custom WAV file to play when listening stops (replaces the default chime). Omit path to reset to default.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to a WAV file" }
                    }
                }
            },
            {
                "name": "play_sound",
                "description": "Play a WAV file through the speakers. Useful for previewing sounds.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to a WAV file to play" }
                    },
                    "required": ["path"]
                }
            }
        ]
    }))
}

fn handle_tools_call(
    session: &mut Session,
    stdout: &mut io::Stdout,
    params: Value,
    id: Value,
) -> Response {
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

    let result = match name {
        "speak" => voice_speak(session, stdout, arguments),
        "listen" => voice_listen(session, arguments),
        "converse" => voice_converse(session, stdout, arguments),
        "cancel" => voice_cancel(),
        "set_voice" => voice_set_voice(session, arguments),
        "set_speed" => voice_set_speed(session, arguments),
        "list_voices" => voice_list_voices(session, arguments),
        "set_start_sound" => voice_set_start_sound(arguments),
        "set_stop_sound" => voice_set_stop_sound(arguments),
        "play_sound" => voice_play_sound(arguments),
        _ => {
            return Response::success(
                id,
                serde_json::json!({
                    "isError": true,
                    "content": [{ "type": "text", "text": format!("Unknown tool: {name}") }]
                }),
            );
        }
    };

    match result {
        Ok(value) => Response::success(
            id,
            serde_json::json!({
                "content": [{ "type": "text", "text": serde_json::to_string(&value).unwrap() }]
            }),
        ),
        Err(e) => Response::success(
            id,
            serde_json::json!({
                "isError": true,
                "content": [{ "type": "text", "text": e.message }]
            }),
        ),
    }
}

// ── Error helper ──────────────────────────────────────────────────────

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

// ── Voice tool handlers ───────────────────────────────────────────────

fn voice_cancel() -> Result<Value, RpcErr> {
    INTERRUPTED.store(true, Ordering::SeqCst);
    Ok(serde_json::json!({"cancelled": true}))
}

fn voice_listen(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
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

fn voice_speak(
    session: &mut Session,
    stdout: &mut io::Stdout,
    params: Value,
) -> Result<Value, RpcErr> {
    let p: SpeakParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    let speed = p.speed.unwrap_or(session.speed);

    let voice_ref: *const voice_tts::Array = if let Some(ref name) = p.voice {
        session.get_voice(name).map_err(RpcErr::invalid_params)? as *const _
    } else {
        &session.voice as *const _
    };
    let voice = unsafe { &*voice_ref };

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

    let started = Instant::now();
    stream_chunks(session, stdout, voice, &chunks, speed)?;
    let duration_ms = started.elapsed().as_millis() as u64;

    Ok(serde_json::json!({
        "duration_ms": duration_ms,
        "chunks": chunks.len(),
    }))
}

fn voice_converse(
    session: &mut Session,
    stdout: &mut io::Stdout,
    params: Value,
) -> Result<Value, RpcErr> {
    let p: ConverseParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    // Speak first
    let speak_params = SpeakParams {
        text: p.text,
        phonemes: p.phonemes,
        voice: p.voice,
        speed: p.speed,
        markdown: p.markdown,
    };
    let speak_result = voice_speak(
        session,
        stdout,
        serde_json::to_value(&speak_params).unwrap_or(Value::Null),
    );

    // If speak fails, still try to get the error but don't abort listen
    let spoke = match speak_result {
        Ok(v) => v,
        Err(e) => serde_json::json!({"error": e.message}),
    };

    // Then listen
    let listen_params = serde_json::json!({
        "max_duration_ms": p.max_duration_ms,
        "silence_timeout_ms": p.silence_timeout_ms,
        "silence_threshold": p.silence_threshold,
        "noise_multiplier": p.noise_multiplier,
        "calibration_ms": p.calibration_ms,
    });
    let heard = match voice_listen(session, listen_params) {
        Ok(v) => v,
        Err(e) => serde_json::json!({"error": e.message}),
    };

    Ok(serde_json::json!({
        "spoke": spoke,
        "heard": heard,
    }))
}

fn voice_set_voice(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
    let p: SetVoiceParams =
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?;

    let voice = session
        .get_voice(&p.voice)
        .map_err(RpcErr::invalid_params)?
        .clone();

    session.voice = voice;
    session.voice_name = p.voice.clone();

    Ok(serde_json::json!({ "voice": p.voice }))
}

fn voice_set_speed(session: &mut Session, params: Value) -> Result<Value, RpcErr> {
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

fn voice_list_voices(session: &Session, params: Value) -> Result<Value, RpcErr> {
    let show_all = params.get("all").and_then(|v| v.as_bool()).unwrap_or(false);

    if show_all {
        // Full catalog with availability status
        let voices: Vec<Value> = voice_tts::catalog::ALL_VOICES
            .iter()
            .map(|v| {
                let builtin = voice_tts::catalog::is_builtin(v.id);
                let cached = builtin || voice_tts::catalog::is_cached(v.id, Some(&session.repo_id));
                let status = if builtin {
                    "builtin"
                } else if cached {
                    "cached"
                } else {
                    "available"
                };
                serde_json::json!({
                    "id": v.id,
                    "name": v.name,
                    "language": v.language,
                    "gender": v.gender,
                    "grade": v.grade,
                    "traits": v.traits,
                    "status": status,
                })
            })
            .collect();
        Ok(serde_json::json!({ "voices": voices }))
    } else {
        // Quick: just builtin voices with metadata
        let voices: Vec<Value> = voice_tts::catalog::ALL_VOICES
            .iter()
            .filter(|v| voice_tts::catalog::is_builtin(v.id))
            .map(|v| {
                serde_json::json!({
                    "id": v.id,
                    "name": v.name,
                    "language": v.language,
                    "gender": v.gender,
                    "grade": v.grade,
                    "traits": v.traits,
                    "status": "builtin",
                })
            })
            .collect();
        Ok(serde_json::json!({ "voices": voices }))
    }
}

fn voice_set_start_sound(params: Value) -> Result<Value, RpcErr> {
    let p: SetSoundParams = if params.is_null() {
        SetSoundParams { path: None }
    } else {
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?
    };

    match p.path {
        Some(path) => {
            let sound = listen::load_wav_sound(std::path::Path::new(&path))
                .map_err(RpcErr::invalid_params)?;
            listen::set_start_sound(Some(sound));
            Ok(serde_json::json!({ "start_sound": path }))
        }
        None => {
            listen::set_start_sound(None);
            Ok(serde_json::json!({ "start_sound": null }))
        }
    }
}

fn voice_set_stop_sound(params: Value) -> Result<Value, RpcErr> {
    let p: SetSoundParams = if params.is_null() {
        SetSoundParams { path: None }
    } else {
        serde_json::from_value(params).map_err(|e| RpcErr::invalid_params(e.to_string()))?
    };

    match p.path {
        Some(path) => {
            let sound = listen::load_wav_sound(std::path::Path::new(&path))
                .map_err(RpcErr::invalid_params)?;
            listen::set_stop_sound(Some(sound));
            Ok(serde_json::json!({ "stop_sound": path }))
        }
        None => {
            listen::set_stop_sound(None);
            Ok(serde_json::json!({ "stop_sound": null }))
        }
    }
}

fn voice_play_sound(params: Value) -> Result<Value, RpcErr> {
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcErr::invalid_params("missing 'path'"))?;

    let sound =
        listen::load_wav_sound(std::path::Path::new(path)).map_err(RpcErr::invalid_params)?;

    listen::play_cached_sound(&sound).map_err(RpcErr::internal)?;

    Ok(serde_json::json!({ "played": path }))
}

// ── Audio playback ────────────────────────────────────────────────────

fn stream_chunks(
    session: &mut Session,
    _stdout: &mut io::Stdout,
    voice: &voice_tts::Array,
    chunks: &[String],
    speed: f32,
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

    while !player.empty() {
        if interrupted() {
            player.stop();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

// ── IO ────────────────────────────────────────────────────────────────

fn write_response(stdout: &mut io::Stdout, resp: &Response) {
    let json = serde_json::to_string(resp).unwrap();
    let _ = writeln!(stdout, "{json}");
    let _ = stdout.flush();
}
