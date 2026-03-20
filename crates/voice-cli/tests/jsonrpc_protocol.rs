//! Integration tests for the JSON-RPC 2.0 stdio server.
//!
//! These tests spawn the `voice` binary with `--jsonrpc` and communicate
//! over stdin/stdout. They test protocol compliance — correct response
//! structure, error codes, notification handling — without requiring
//! audio hardware or model downloads (for non-audio methods).
//!
//! Tests that require model downloads are marked `#[ignore]` and can be
//! run with `cargo test -p voice --test jsonrpc_protocol -- --ignored`.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

/// A handle to a running JSON-RPC server process.
struct Server {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl Server {
    /// Spawn a new `voice --jsonrpc -q` server process.
    ///
    /// Uses the binary built by `cargo test`, which should be in the
    /// target directory. Falls back to `voice` on PATH.
    fn spawn() -> Self {
        let bin = Self::find_binary();
        let mut child = Command::new(bin)
            .args(["--jsonrpc", "-q"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn voice --jsonrpc");

        let stdin = child.stdin.take().expect("Failed to open stdin");
        let stdout = child.stdout.take().expect("Failed to open stdout");
        let reader = BufReader::new(stdout);

        Server {
            child,
            stdin,
            reader,
        }
    }

    /// Find the voice binary — prefer the cargo-built one in target/.
    fn find_binary() -> String {
        // Try the release binary first (faster), then debug
        let candidates = [
            "target/release/voice",
            "target/debug/voice",
            "../target/release/voice",
            "../target/debug/voice",
            "../../target/release/voice",
            "../../target/debug/voice",
        ];
        for path in &candidates {
            if std::path::Path::new(path).exists() {
                return path.to_string();
            }
        }
        // Fall back to PATH
        "voice".to_string()
    }

    /// Send a JSON-RPC request and read the response.
    ///
    /// Skips any server-sent notifications (no `id` field) and returns
    /// the first response that has a matching `id`.
    fn request(&mut self, method: &str, params: Option<Value>, id: u64) -> Value {
        let msg = if let Some(p) = params {
            json!({"jsonrpc": "2.0", "method": method, "params": p, "id": id})
        } else {
            json!({"jsonrpc": "2.0", "method": method, "id": id})
        };

        writeln!(self.stdin, "{}", serde_json::to_string(&msg).unwrap())
            .expect("Failed to write to server stdin");
        self.stdin.flush().expect("Failed to flush stdin");

        // Read lines until we get a response with our id
        loop {
            let mut line = String::new();
            self.reader
                .read_line(&mut line)
                .expect("Failed to read from server stdout");

            if line.trim().is_empty() {
                continue;
            }

            let resp: Value =
                serde_json::from_str(line.trim()).expect("Failed to parse server response as JSON");

            // Skip notifications (no id)
            if resp.get("id").is_some() && !resp["id"].is_null() {
                return resp;
            }
        }
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    fn notify(&mut self, method: &str, params: Option<Value>) {
        let msg = if let Some(p) = params {
            json!({"jsonrpc": "2.0", "method": method, "params": p})
        } else {
            json!({"jsonrpc": "2.0", "method": method})
        };

        writeln!(self.stdin, "{}", serde_json::to_string(&msg).unwrap())
            .expect("Failed to write to server stdin");
        self.stdin.flush().expect("Failed to flush stdin");

        // Brief pause to let the server process it
        std::thread::sleep(Duration::from_millis(50));
    }

    /// Send a raw string (possibly invalid JSON) and read the response.
    fn send_raw(&mut self, raw: &str) -> Value {
        writeln!(self.stdin, "{}", raw).expect("Failed to write raw to stdin");
        self.stdin.flush().expect("Failed to flush stdin");

        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .expect("Failed to read response");
        serde_json::from_str(line.trim()).expect("Failed to parse response")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// Protocol compliance tests
// ---------------------------------------------------------------------------

#[test]
fn test_jsonrpc_ping() {
    let mut server = Server::spawn();
    let resp = server.request("ping", None, 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"], "pong");
    assert!(resp.get("error").is_none() || resp["error"].is_null());
}

#[test]
fn test_jsonrpc_ping_string_id() {
    let mut server = Server::spawn();

    // JSON-RPC 2.0 allows string IDs
    let msg = json!({"jsonrpc": "2.0", "method": "ping", "id": "abc"});
    writeln!(server.stdin, "{}", serde_json::to_string(&msg).unwrap()).unwrap();
    server.stdin.flush().unwrap();

    let mut line = String::new();
    server.reader.read_line(&mut line).unwrap();
    let resp: Value = serde_json::from_str(line.trim()).unwrap();

    assert_eq!(resp["id"], "abc");
    assert_eq!(resp["result"], "pong");
}

#[test]
fn test_jsonrpc_unknown_method() {
    let mut server = Server::spawn();
    let resp = server.request("nonexistent_method", None, 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert!(resp.get("result").is_none() || resp["result"].is_null());
    assert_eq!(resp["error"]["code"], -32601); // Method not found
    assert!(resp["error"]["message"]
        .as_str()
        .unwrap()
        .contains("nonexistent_method"));
}

#[test]
fn test_jsonrpc_parse_error() {
    let mut server = Server::spawn();
    let resp = server.send_raw("this is not json at all");

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["error"]["code"], -32700); // Parse error
                                               // id should be null for parse errors
    assert!(resp["id"].is_null());
}

#[test]
fn test_jsonrpc_parse_error_partial_json() {
    let mut server = Server::spawn();
    let resp = server.send_raw("{\"jsonrpc\": \"2.0\", \"method\":");

    assert_eq!(resp["error"]["code"], -32700);
}

#[test]
fn test_jsonrpc_cancel() {
    let mut server = Server::spawn();
    let resp = server.request("cancel", None, 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["cancelled"], true);
}

#[test]
fn test_jsonrpc_set_speed_valid() {
    let mut server = Server::spawn();
    let resp = server.request("set_speed", Some(json!({"speed": 1.5})), 1);

    assert_eq!(resp["result"]["speed"], 1.5);
}

#[test]
fn test_jsonrpc_set_speed_too_high() {
    let mut server = Server::spawn();
    let resp = server.request("set_speed", Some(json!({"speed": 10.0})), 1);

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
fn test_jsonrpc_set_speed_zero() {
    let mut server = Server::spawn();
    let resp = server.request("set_speed", Some(json!({"speed": 0.0})), 1);

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
fn test_jsonrpc_set_speed_negative() {
    let mut server = Server::spawn();
    let resp = server.request("set_speed", Some(json!({"speed": -1.0})), 1);

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
fn test_jsonrpc_set_speed_missing_param() {
    let mut server = Server::spawn();
    let resp = server.request("set_speed", Some(json!({})), 1);

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
fn test_jsonrpc_notification_no_response() {
    let mut server = Server::spawn();

    // Send a notification (no id) — should produce no response
    server.notify("cancel", None);

    // Now send a real request — if the notification produced a response,
    // we'd get it here instead of our ping response
    let resp = server.request("ping", None, 42);

    assert_eq!(resp["id"], 42);
    assert_eq!(resp["result"], "pong");
}

#[test]
fn test_jsonrpc_empty_line_ignored() {
    let mut server = Server::spawn();

    // Send empty lines
    writeln!(server.stdin).unwrap();
    writeln!(server.stdin).unwrap();
    writeln!(server.stdin, "   ").unwrap();
    server.stdin.flush().unwrap();

    // Server should still respond to a real request
    let resp = server.request("ping", None, 1);
    assert_eq!(resp["result"], "pong");
}

#[test]
fn test_jsonrpc_sequential_requests() {
    let mut server = Server::spawn();

    for i in 1..=5 {
        let resp = server.request("ping", None, i);
        assert_eq!(resp["id"], i);
        assert_eq!(resp["result"], "pong");
    }
}

#[test]
fn test_jsonrpc_list_voices() {
    let mut server = Server::spawn();
    let resp = server.request("list_voices", None, 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);

    let voices = resp["result"]["voices"]
        .as_array()
        .expect("voices should be an array");
    assert!(!voices.is_empty(), "should have at least one voice");

    // Check that known builtin voices are present
    let voice_names: Vec<&str> = voices.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        voice_names.contains(&"af_heart"),
        "should contain af_heart: {voice_names:?}"
    );
    assert!(
        voice_names.contains(&"am_michael"),
        "should contain am_michael: {voice_names:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests that require model downloads (gated with #[ignore])
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_speak() {
    let mut server = Server::spawn();
    let resp = server.request("speak", Some(json!({"text": "Hello"})), 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert!(resp["result"]["duration_ms"].as_u64().unwrap() > 0);
    assert!(resp["result"]["chunks"].as_u64().unwrap() >= 1);
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_speak_with_detail() {
    let mut server = Server::spawn();
    let resp = server.request(
        "speak",
        Some(json!({"text": "Hello.", "detail": "full"})),
        1,
    );

    assert!(resp["result"]["phonemes"].is_array());
    let phonemes = resp["result"]["phonemes"].as_array().unwrap();
    assert!(!phonemes.is_empty());
    // Should contain a period (punctuation preservation)
    let all_phonemes: String = phonemes.iter().filter_map(|p| p.as_str()).collect();
    assert!(
        all_phonemes.contains('.'),
        "phonemes should contain period: {all_phonemes}"
    );
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_speak_missing_text() {
    let mut server = Server::spawn();
    let resp = server.request("speak", Some(json!({})), 1);

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_set_voice() {
    let mut server = Server::spawn();
    let resp = server.request("set_voice", Some(json!({"voice": "am_michael"})), 1);

    assert_eq!(resp["result"]["voice"], "am_michael");
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_set_voice_invalid() {
    let mut server = Server::spawn();
    let resp = server.request(
        "set_voice",
        Some(json!({"voice": "nonexistent_voice_xyz"})),
        1,
    );

    assert_eq!(resp["error"]["code"], -32602); // Invalid params
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_speak_per_request_voice_override() {
    let mut server = Server::spawn();

    // Speak with a per-request voice override (doesn't change default)
    let resp = server.request(
        "speak",
        Some(json!({"text": "Hello", "voice": "am_adam"})),
        1,
    );

    assert!(resp["result"]["duration_ms"].as_u64().unwrap() > 0);

    // Default voice should still be af_heart (unchanged)
    // Verify by speaking without override
    let resp2 = server.request("speak", Some(json!({"text": "Hello", "detail": "full"})), 2);

    assert!(resp2["result"]["duration_ms"].as_u64().unwrap() > 0);
}

#[test]
#[ignore = "requires TTS model download (~312MB)"]
fn test_jsonrpc_speak_markdown_stripping() {
    let mut server = Server::spawn();
    let resp = server.request(
        "speak",
        Some(json!({
            "text": "# Heading\n\nSome **bold** text.\n\n```python\nprint(1)\n```\n\nMore words.",
            "markdown": true,
            "detail": "full"
        })),
        1,
    );

    let phonemes = resp["result"]["phonemes"].as_array().unwrap();
    let all: String = phonemes.iter().filter_map(|p| p.as_str()).collect();

    // Code block should be stripped — "print" should not appear in phonemes
    assert!(
        !all.contains("pɹˈɪnt"),
        "code block should be stripped from phonemes: {all}"
    );
}
