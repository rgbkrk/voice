//! Synchronous client for talking to the voice daemon.
//!
//! The MCP server is synchronous (no tokio runtime), so this client
//! uses std::os::unix::net::UnixStream directly.

use crate::frames::{read_frame_sync, write_frame_sync, Frame, FrameType};
use crate::rpc::{self, Response};
use serde_json::Value;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;
use voice_stream::TtsStreamEvent;

const SOCKET_ENV: &str = "VOICE_DAEMON_SOCKET";
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(120);
const DURATION_CALL_TIMEOUT_PAD: Duration = Duration::from_secs(120);
const VOXTRAL_TTS_READ_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Optional TTS engine controls for daemon requests.
#[derive(Debug, Clone, Default)]
pub struct TtsRequestOptions<'a> {
    pub engine: Option<&'a str>,
    pub voxtral_model: Option<&'a str>,
    pub voxtral_max_frames: Option<usize>,
    pub voxtral_flow_steps: Option<usize>,
    pub voxtral_stream_begin_frames: Option<usize>,
    pub voxtral_kv_cache: bool,
}

/// Options for daemon streaming TTS requests.
#[derive(Debug, Clone, Default)]
pub struct StreamSpeakOptions<'a> {
    pub voice: Option<&'a str>,
    pub speed: Option<f64>,
    pub sample_rate: Option<u32>,
    pub frame_ms: Option<u32>,
    pub tts: TtsRequestOptions<'a>,
}

/// A synchronous client connection to the voice daemon.
pub struct DaemonClient {
    stream: UnixStream,
}

/// Get the daemon socket path.
pub fn daemon_socket_path() -> PathBuf {
    if let Ok(path) = std::env::var(SOCKET_ENV) {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }

    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    dir.join("daemon.sock")
}

/// Environment variable that overrides the daemon socket path.
pub fn daemon_socket_env_var() -> &'static str {
    SOCKET_ENV
}

/// Check if the daemon is running (socket exists and accepts connections).
pub fn daemon_is_running() -> bool {
    let path = daemon_socket_path();
    if !path.exists() {
        return false;
    }
    UnixStream::connect(&path).is_ok()
}

impl DaemonClient {
    /// Connect to the daemon. Returns None if daemon isn't running.
    pub fn connect() -> Option<Self> {
        let path = daemon_socket_path();
        let stream = UnixStream::connect(&path).ok()?;
        stream.set_read_timeout(Some(DEFAULT_READ_TIMEOUT)).ok()?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .ok()?;
        Some(Self { stream })
    }

    /// Send a JSON-RPC request and get the response.
    pub fn call(&mut self, method: &str, params: Value) -> Result<Response, String> {
        self.call_with_read_timeout(method, params, None)
    }

    fn call_with_read_timeout(
        &mut self,
        method: &str,
        params: Value,
        read_timeout: Option<Duration>,
    ) -> Result<Response, String> {
        if let Some(timeout) = read_timeout {
            self.stream
                .set_read_timeout(Some(timeout))
                .map_err(|e| format!("set read timeout: {}", e))?;
        }

        self.call_inner(method, params)
    }

    fn call_inner(&mut self, method: &str, params: Value) -> Result<Response, String> {
        let req = rpc::Request::new(method, params).with_id(1);
        let json = serde_json::to_vec(&req).map_err(|e| format!("serialize: {}", e))?;

        write_frame_sync(&mut self.stream, &Frame::request(&json))
            .map_err(|e| format!("write frame: {}", e))?;

        let frame = read_frame_sync(&mut self.stream)
            .map_err(|e| format!("read frame: {}", e))?
            .ok_or_else(|| "connection closed before response".to_string())?;

        if frame.frame_type != FrameType::Response {
            return Err(format!(
                "unexpected frame type {:?}; expected response",
                frame.frame_type
            ));
        }

        frame
            .json::<Response>()
            .map_err(|e| format!("parse response: {}", e))
    }

    /// Convenience: send a speak request. Returns immediately with queue_id (fire-and-forget).
    pub fn speak(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f64>,
    ) -> Result<Response, String> {
        self.speak_with_options(text, voice, speed, TtsRequestOptions::default())
    }

    /// Convenience: send a speak request with explicit TTS engine controls.
    pub fn speak_with_options(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f64>,
        options: TtsRequestOptions<'_>,
    ) -> Result<Response, String> {
        self.speak_with_options_and_wait(text, voice, speed, options, false)
    }

    /// Convenience: send a speak request and optionally wait for playback completion.
    pub fn speak_with_options_and_wait(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f64>,
        options: TtsRequestOptions<'_>,
        wait: bool,
    ) -> Result<Response, String> {
        let read_timeout = wait
            .then(|| read_timeout_for_tts_options(&options))
            .flatten();
        let mut params = serde_json::json!({"text": text, "wait": wait});
        if let Some(v) = voice {
            params["voice"] = Value::String(v.to_string());
        }
        if let Some(s) = speed {
            params["speed"] = serde_json::json!(s);
        }
        insert_tts_options(&mut params, options);
        self.call_with_read_timeout("speak", params, read_timeout)
    }

    /// Convenience: synthesize with explicit TTS engine controls.
    pub fn synthesize_with_format_and_options(
        &mut self,
        text: &str,
        output_path: &str,
        output_format: Option<&str>,
        voice: Option<&str>,
        speed: Option<f64>,
        options: TtsRequestOptions<'_>,
    ) -> Result<Response, String> {
        let mut params = serde_json::json!({
            "text": text,
            "output_path": output_path,
            "wait": true,
        });
        if let Some(format) = output_format {
            params["format"] = Value::String(format.to_string());
        }
        if let Some(v) = voice {
            params["voice"] = Value::String(v.to_string());
        }
        if let Some(s) = speed {
            params["speed"] = serde_json::json!(s);
        }
        let read_timeout = read_timeout_for_tts_options(&options);
        insert_tts_options(&mut params, options);
        self.call_with_read_timeout("synthesize", params, read_timeout)
    }

    /// Stream TTS audio from the daemon with explicit TTS engine controls.
    pub fn stream_speak_with_options<F>(
        &mut self,
        text: &str,
        options: StreamSpeakOptions<'_>,
        on_event: F,
    ) -> Result<Response, String>
    where
        F: FnMut(TtsStreamEvent) -> Result<(), String>,
    {
        self.stream_speak_with_options_observed(text, options, |_| Ok(()), on_event)
    }

    /// Stream TTS audio from the daemon while observing the initial queued response.
    pub fn stream_speak_with_options_observed<R, F>(
        &mut self,
        text: &str,
        options: StreamSpeakOptions<'_>,
        mut on_response: R,
        mut on_event: F,
    ) -> Result<Response, String>
    where
        R: FnMut(&Response) -> Result<(), String>,
        F: FnMut(TtsStreamEvent) -> Result<(), String>,
    {
        let mut params = serde_json::json!({ "text": text });
        if let Some(v) = options.voice {
            params["voice"] = Value::String(v.to_string());
        }
        if let Some(s) = options.speed {
            params["speed"] = serde_json::json!(s);
        }
        if let Some(rate) = options.sample_rate {
            params["sample_rate"] = serde_json::json!(rate);
        }
        if let Some(ms) = options.frame_ms {
            params["frame_ms"] = serde_json::json!(ms);
        }
        let read_timeout = read_timeout_for_tts_options(&options.tts);
        insert_tts_options(&mut params, options.tts);
        if let Some(timeout) = read_timeout {
            self.stream
                .set_read_timeout(Some(timeout))
                .map_err(|e| format!("set read timeout: {}", e))?;
        }

        let req = rpc::Request::new("stream_speak", params).with_id(1);
        let json = serde_json::to_vec(&req).map_err(|e| format!("serialize: {}", e))?;

        write_frame_sync(&mut self.stream, &Frame::request(&json))
            .map_err(|e| format!("write frame: {}", e))?;

        let frame = read_frame_sync(&mut self.stream)
            .map_err(|e| format!("read frame: {}", e))?
            .ok_or_else(|| "connection closed before stream response".to_string())?;
        if frame.frame_type != FrameType::Response {
            return Err(format!(
                "unexpected frame type {:?}; expected stream response",
                frame.frame_type
            ));
        }

        let response = frame
            .json::<Response>()
            .map_err(|e| format!("parse response: {}", e))?;
        on_response(&response)?;
        if response.error.is_some() {
            return Ok(response);
        }

        loop {
            let frame = read_frame_sync(&mut self.stream)
                .map_err(|e| format!("read stream frame: {}", e))?
                .ok_or_else(|| "connection closed before stream finished".to_string())?;

            if frame.frame_type != FrameType::Event {
                return Err(format!(
                    "unexpected frame type {:?}; expected stream event",
                    frame.frame_type
                ));
            }

            let event = frame
                .json::<rpc::Event>()
                .map_err(|e| format!("parse event envelope: {}", e))?;
            let stream_event: TtsStreamEvent = serde_json::from_value(event.data)
                .map_err(|e| format!("parse stream event: {}", e))?;
            let terminal = stream_event.is_terminal();
            on_event(stream_event)?;
            if terminal {
                break;
            }
        }

        Ok(response)
    }

    /// Stream caller-supplied PCM frames into the daemon for transcription.
    ///
    /// This is an ingestion contract for WebRTC/bridge clients. The daemon
    /// receives `stt.audio` event frames, then transcribes after `stt.end` and
    /// emits either `stt.transcribed` or `stt.error`.
    pub fn stream_transcribe<F>(
        &mut self,
        frames: &[Vec<i16>],
        sample_rate: u32,
        frame_ms: u32,
        mut on_event: F,
    ) -> Result<Response, String>
    where
        F: FnMut(rpc::Event) -> Result<(), String>,
    {
        let params = serde_json::json!({
            "sample_rate": sample_rate,
            "channels": 1,
            "encoding": "pcm_s16le",
            "frame_ms": frame_ms,
        });
        let req = rpc::Request::new("stream_transcribe", params).with_id(1);
        let json = serde_json::to_vec(&req).map_err(|e| format!("serialize: {}", e))?;

        write_frame_sync(&mut self.stream, &Frame::request(&json))
            .map_err(|e| format!("write frame: {}", e))?;

        let frame = read_frame_sync(&mut self.stream)
            .map_err(|e| format!("read frame: {}", e))?
            .ok_or_else(|| "connection closed before stream_transcribe response".to_string())?;
        if frame.frame_type != FrameType::Response {
            return Err(format!(
                "unexpected frame type {:?}; expected stream_transcribe response",
                frame.frame_type
            ));
        }

        let response = frame
            .json::<Response>()
            .map_err(|e| format!("parse response: {}", e))?;
        if response.error.is_some() {
            return Ok(response);
        }
        let stream_id = response
            .result
            .as_ref()
            .and_then(|result| result.get("stream_id"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| "stream_transcribe response missing stream_id".to_string())?
            .to_string();

        for (sequence, samples) in frames.iter().enumerate() {
            let envelope = rpc::Event::new(
                "stt.audio",
                serde_json::json!({
                    "frame": {
                        "stream_id": &stream_id,
                        "sequence": sequence as u64,
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                        "frame_ms": frame_ms,
                        "sample_count": samples.len(),
                        "samples": samples,
                    }
                }),
            );
            let json =
                serde_json::to_vec(&envelope).map_err(|e| format!("serialize event: {e}"))?;
            write_frame_sync(&mut self.stream, &Frame::event(&json))
                .map_err(|e| format!("write stt.audio: {e}"))?;
        }

        let envelope = rpc::Event::new("stt.end", serde_json::json!({ "stream_id": &stream_id }));
        let json = serde_json::to_vec(&envelope).map_err(|e| format!("serialize end: {e}"))?;
        write_frame_sync(&mut self.stream, &Frame::event(&json))
            .map_err(|e| format!("write stt.end: {e}"))?;

        loop {
            let frame = read_frame_sync(&mut self.stream)
                .map_err(|e| format!("read stream_transcribe frame: {}", e))?
                .ok_or_else(|| "connection closed before stream_transcribe finished".to_string())?;

            if frame.frame_type != FrameType::Event {
                return Err(format!(
                    "unexpected frame type {:?}; expected stream_transcribe event",
                    frame.frame_type
                ));
            }

            let event = frame
                .json::<rpc::Event>()
                .map_err(|e| format!("parse event envelope: {}", e))?;
            let terminal = matches!(event.event.as_str(), "stt.transcribed" | "stt.error");
            on_event(event)?;
            if terminal {
                break;
            }
        }

        Ok(response)
    }

    /// Convenience: send a listen request. Blocks until transcription completes.
    pub fn listen(&mut self, max_duration_ms: Option<u64>) -> Result<Response, String> {
        let mut params = serde_json::json!({"wait": true});
        if let Some(ms) = max_duration_ms {
            params["max_duration_ms"] = serde_json::json!(ms);
        }
        self.call_with_read_timeout(
            "listen",
            params,
            max_duration_ms.map(read_timeout_for_max_duration),
        )
    }

    /// Start a converse: speak the prompt then capture the user's spoken reply.
    ///
    /// Returns immediately (does not wait for the human) with a `converse_id`.
    /// Fetch the transcript later with [`converse_result`](Self::converse_result).
    pub fn converse(
        &mut self,
        text: &str,
        voice: Option<&str>,
        max_duration_ms: Option<u64>,
        options: TtsRequestOptions<'_>,
    ) -> Result<Response, String> {
        let mut params = serde_json::json!({ "text": text });
        if let Some(v) = voice {
            params["voice"] = Value::String(v.to_string());
        }
        if let Some(ms) = max_duration_ms {
            params["max_duration_ms"] = serde_json::json!(ms);
        }
        insert_tts_options(&mut params, options);
        self.call("converse", params)
    }

    /// Fetch the state/transcript of a converse by id. With `wait_ms`, the
    /// daemon long-polls up to that long for a terminal result before replying.
    pub fn converse_result(
        &mut self,
        converse_id: &str,
        wait_ms: Option<u64>,
    ) -> Result<Response, String> {
        let mut params = serde_json::json!({ "converse_id": converse_id });
        let read_timeout = wait_ms.map(|ms| {
            params["wait_ms"] = serde_json::json!(ms);
            Duration::from_millis(ms).saturating_add(DURATION_CALL_TIMEOUT_PAD)
        });
        self.call_with_read_timeout("converse_result", params, read_timeout)
    }

    /// Convenience: get daemon status.
    pub fn status(&mut self) -> Result<Response, String> {
        self.call("status", serde_json::json!({}))
    }

    /// Convenience: set a default voice for a specific TTS engine.
    pub fn set_voice_for_engine(&mut self, engine: &str, voice: &str) -> Result<Response, String> {
        self.call(
            "set_voice",
            serde_json::json!({ "engine": engine, "voice": voice }),
        )
    }

    /// Convenience: set the daemon default TTS engine.
    pub fn set_engine(
        &mut self,
        engine: &str,
        voxtral_model: Option<&str>,
    ) -> Result<Response, String> {
        let mut params = serde_json::json!({ "engine": engine });
        if let Some(model) = voxtral_model {
            params["voxtral_model"] = Value::String(model.to_string());
        }
        self.call("set_engine", params)
    }

    /// Convenience: set daemon default speed.
    pub fn set_speed(&mut self, speed: f64) -> Result<Response, String> {
        self.call("set_speed", serde_json::json!({ "speed": speed }))
    }

    /// Convenience: list known voices and the current daemon default.
    pub fn list_voices(&mut self) -> Result<Response, String> {
        self.call("list_voices", serde_json::json!({}))
    }

    /// Convenience: cancel a specific queue item.
    pub fn cancel_item(&mut self, queue_id: &str) -> Result<Response, String> {
        self.call("cancel_item", serde_json::json!({ "queue_id": queue_id }))
    }
}

fn read_timeout_for_max_duration(max_duration_ms: u64) -> Duration {
    Duration::from_millis(max_duration_ms).saturating_add(DURATION_CALL_TIMEOUT_PAD)
}

fn read_timeout_for_tts_options(options: &TtsRequestOptions<'_>) -> Option<Duration> {
    if options.engine == Some("voxtral") || options.voxtral_model.is_some() {
        Some(VOXTRAL_TTS_READ_TIMEOUT)
    } else {
        None
    }
}

fn insert_tts_options(params: &mut Value, options: TtsRequestOptions<'_>) {
    if let Some(engine) = options.engine {
        params["engine"] = Value::String(engine.to_string());
    }
    if let Some(model) = options.voxtral_model {
        params["voxtral_model"] = Value::String(model.to_string());
    }
    if let Some(max_frames) = options.voxtral_max_frames {
        params["voxtral_max_frames"] = serde_json::json!(max_frames);
    }
    if let Some(flow_steps) = options.voxtral_flow_steps {
        params["voxtral_flow_steps"] = serde_json::json!(flow_steps);
    }
    if let Some(stream_begin_frames) = options.voxtral_stream_begin_frames {
        params["voxtral_stream_begin_frames"] = serde_json::json!(stream_begin_frames);
    }
    if options.voxtral_kv_cache {
        params["voxtral_kv_cache"] = Value::Bool(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{read_frame_sync, write_frame_sync, Frame};
    use crate::rpc::{Event, Request, Response};
    use std::os::unix::net::UnixListener;
    use std::sync::Mutex;
    use std::thread;
    use voice_stream::{AudioEncoding, StreamEnded, StreamMetadata, TtsStreamEvent};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn socket_path_uses_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, "/tmp/voice-test.sock");

        assert_eq!(daemon_socket_path(), PathBuf::from("/tmp/voice-test.sock"));

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
    }

    #[test]
    fn call_rejects_non_response_frame() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("voice-protocol-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame_sync(&mut stream).unwrap().unwrap();
            write_frame_sync(&mut stream, &Frame::event(b"{}")).unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let err = client.call("status", serde_json::json!({})).unwrap_err();

        assert!(err.contains("unexpected frame type"));

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn call_parses_response_frame() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-test-ok-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame_sync(&mut stream).unwrap().unwrap();
            let response =
                Response::success(Some(1.into()), serde_json::json!({ "status": "idle" }));
            let payload = serde_json::to_vec(&response).unwrap();
            write_frame_sync(&mut stream, &Frame::response(&payload)).unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let response = client.call("status", serde_json::json!({})).unwrap();
        assert_eq!(response.result.unwrap()["status"], "idle");

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn speak_with_options_and_wait_sends_wait_true() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-speak-wait-test-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let frame = read_frame_sync(&mut stream).unwrap().unwrap();
            let request = frame.json::<Request>().unwrap();
            assert_eq!(request.method, "speak");
            assert_eq!(request.params["text"], "hello");
            assert_eq!(request.params["voice"], "casual_male");
            assert_eq!(request.params["wait"], true);
            assert_eq!(request.params["engine"], "voxtral");
            assert_eq!(request.params["voxtral_kv_cache"], true);
            assert_eq!(request.params["voxtral_stream_begin_frames"], 3);

            let response = Response::success(
                Some(1.into()),
                serde_json::json!({
                    "queue_id": "q",
                    "status": "completed",
                    "result": "{\"engine\":\"voxtral\"}",
                }),
            );
            write_frame_sync(
                &mut stream,
                &Frame::response(&serde_json::to_vec(&response).unwrap()),
            )
            .unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let response = client
            .speak_with_options_and_wait(
                "hello",
                Some("casual_male"),
                Some(1.0),
                TtsRequestOptions {
                    engine: Some("voxtral"),
                    voxtral_stream_begin_frames: Some(3),
                    voxtral_kv_cache: true,
                    ..TtsRequestOptions::default()
                },
                true,
            )
            .unwrap();

        assert!(response.error.is_none());

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn duration_calls_pad_read_timeout() {
        assert_eq!(
            read_timeout_for_max_duration(1_500),
            Duration::from_millis(121_500)
        );
        assert_eq!(
            read_timeout_for_max_duration(180_000),
            Duration::from_secs(300)
        );
    }

    #[test]
    fn voxtral_tts_options_use_extended_read_timeout() {
        assert_eq!(
            read_timeout_for_tts_options(&TtsRequestOptions {
                engine: Some("voxtral"),
                ..TtsRequestOptions::default()
            }),
            Some(VOXTRAL_TTS_READ_TIMEOUT)
        );
        assert_eq!(
            read_timeout_for_tts_options(&TtsRequestOptions {
                voxtral_model: Some("mistralai/Voxtral-4B-TTS-2603"),
                ..TtsRequestOptions::default()
            }),
            Some(VOXTRAL_TTS_READ_TIMEOUT)
        );
        assert_eq!(
            read_timeout_for_tts_options(&TtsRequestOptions::default()),
            None
        );
    }

    #[test]
    fn stream_speak_reads_events_until_terminal() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-stream-test-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame_sync(&mut stream).unwrap().unwrap();

            let response = Response::success(
                Some(serde_json::json!(1)),
                serde_json::json!({
                    "queue_id": "q",
                    "stream_id": "s",
                    "status": "queued",
                }),
            );
            write_frame_sync(
                &mut stream,
                &Frame::response(&serde_json::to_vec(&response).unwrap()),
            )
            .unwrap();

            let started = TtsStreamEvent::Started {
                metadata: StreamMetadata {
                    stream_id: "s".to_string(),
                    sample_rate: 24_000,
                    source_sample_rate: 24_000,
                    channels: 1,
                    encoding: AudioEncoding::PcmS16Le,
                    frame_ms: 20,
                    voice: Some("af_heart".to_string()),
                    speed: 1.0,
                    total_phoneme_chunks: 1,
                },
            };
            let ended = TtsStreamEvent::Ended(StreamEnded {
                stream_id: "s".to_string(),
                frames: 0,
                samples: 0,
                duration_ms: 0,
                elapsed_ms: 1,
            });

            for event in [started, ended] {
                let envelope =
                    Event::new(event.event_name(), serde_json::to_value(&event).unwrap());
                write_frame_sync(
                    &mut stream,
                    &Frame::event(&serde_json::to_vec(&envelope).unwrap()),
                )
                .unwrap();
            }
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let mut events = Vec::new();
        let response = client
            .stream_speak_with_options("hello", StreamSpeakOptions::default(), |event| {
                events.push(event.event_name().to_string());
                Ok(())
            })
            .unwrap();

        assert!(response.error.is_none());
        assert_eq!(events, vec!["tts.started", "tts.ended"]);

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn stream_speak_observer_sees_initial_response_before_events() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-stream-observer-test-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame_sync(&mut stream).unwrap().unwrap();

            let response = Response::success(
                Some(serde_json::json!(1)),
                serde_json::json!({
                    "queue_id": "q-observed",
                    "stream_id": "s-observed",
                    "status": "queued",
                }),
            );
            write_frame_sync(
                &mut stream,
                &Frame::response(&serde_json::to_vec(&response).unwrap()),
            )
            .unwrap();

            let ended = TtsStreamEvent::Ended(StreamEnded {
                stream_id: "s-observed".to_string(),
                frames: 0,
                samples: 0,
                duration_ms: 0,
                elapsed_ms: 1,
            });
            let envelope = Event::new(ended.event_name(), serde_json::to_value(&ended).unwrap());
            write_frame_sync(
                &mut stream,
                &Frame::event(&serde_json::to_vec(&envelope).unwrap()),
            )
            .unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let observed = std::cell::RefCell::new(Vec::new());
        let response = client
            .stream_speak_with_options_observed(
                "hello",
                StreamSpeakOptions::default(),
                |response| {
                    let queue_id = response
                        .result
                        .as_ref()
                        .and_then(|result| result.get("queue_id"))
                        .and_then(|queue_id| queue_id.as_str())
                        .unwrap();
                    observed.borrow_mut().push(format!("response:{queue_id}"));
                    Ok(())
                },
                |event| {
                    observed.borrow_mut().push(event.event_name().to_string());
                    Ok(())
                },
            )
            .unwrap();

        assert!(response.error.is_none());
        assert_eq!(
            observed.into_inner(),
            vec!["response:q-observed", "tts.ended"]
        );

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn stream_transcribe_writes_audio_events_and_reads_terminal_event() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-stream-transcribe-test-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request_frame = read_frame_sync(&mut stream).unwrap().unwrap();
            let request = request_frame.json::<rpc::Request>().unwrap();
            assert_eq!(request.method, "stream_transcribe");
            assert_eq!(request.params["sample_rate"], 48_000);

            let response = Response::success(
                Some(serde_json::json!(1)),
                serde_json::json!({
                    "stream_id": "stt-stream",
                    "status": "receiving",
                    "sample_rate": 48_000,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "frame_ms": 20,
                }),
            );
            write_frame_sync(
                &mut stream,
                &Frame::response(&serde_json::to_vec(&response).unwrap()),
            )
            .unwrap();

            let audio_frame = read_frame_sync(&mut stream).unwrap().unwrap();
            assert_eq!(audio_frame.frame_type, FrameType::Event);
            let audio_event = audio_frame.json::<Event>().unwrap();
            assert_eq!(audio_event.event, "stt.audio");
            assert_eq!(audio_event.data["frame"]["stream_id"], "stt-stream");
            assert_eq!(
                audio_event.data["frame"]["samples"],
                serde_json::json!([0, 1, -1])
            );

            let end_frame = read_frame_sync(&mut stream).unwrap().unwrap();
            assert_eq!(end_frame.frame_type, FrameType::Event);
            let end_event = end_frame.json::<Event>().unwrap();
            assert_eq!(end_event.event, "stt.end");
            assert_eq!(end_event.data["stream_id"], "stt-stream");

            let terminal = Event::new(
                "stt.transcribed",
                serde_json::json!({
                    "stream_id": "stt-stream",
                    "text": "hello",
                    "tokens": 1,
                }),
            );
            write_frame_sync(
                &mut stream,
                &Frame::event(&serde_json::to_vec(&terminal).unwrap()),
            )
            .unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let mut events = Vec::new();
        let response = client
            .stream_transcribe(&[vec![0, 1, -1]], 48_000, 20, |event| {
                events.push(event);
                Ok(())
            })
            .unwrap();

        assert!(response.error.is_none());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "stt.transcribed");
        assert_eq!(events[0].data["text"], "hello");

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }
}
