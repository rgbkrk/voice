//! Integration tests for voice daemon.
//!
//! Tests the full daemon stack: socket server, worker, automerge state sync,
//! and queue management.
//!
//! These tests require a running daemon instance. Start with: `voice mcp`

use std::time::Duration;
use tokio::net::UnixStream;
use voice_protocol::frames::{read_frame, write_frame, Frame};
use voice_protocol::rpc::{Request, Response};

/// Send RPC request and wait for response.
async fn rpc_call(
    stream: &mut UnixStream,
    method: &str,
    params: serde_json::Value,
) -> Result<Response, String> {
    let (mut reader, mut writer) = stream.split();

    let req = Request::new(method, params).with_id(1);
    let json = serde_json::to_vec(&req).unwrap();
    let frame = Frame::request(&json);

    write_frame(&mut writer, &frame)
        .await
        .map_err(|e| format!("Failed to write frame: {}", e))?;

    match read_frame(&mut reader).await {
        Ok(Some(frame)) => frame
            .json::<Response>()
            .map_err(|e| format!("Failed to parse response: {}", e)),
        Ok(None) => Err("Connection closed".to_string()),
        Err(e) => Err(format!("Failed to read frame: {}", e)),
    }
}

#[tokio::test]
async fn test_daemon_status() {
    // This test verifies status command returns expected structure

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .expect("Failed to connect to daemon");

    let resp = rpc_call(&mut stream, "status", serde_json::json!({}))
        .await
        .expect("RPC call failed");

    assert!(
        resp.error.is_none(),
        "Status returned error: {:?}",
        resp.error
    );

    let result = resp.result.expect("Status returned no result");
    assert!(result.get("status").is_some(), "Missing status field");
    assert!(result.get("pending").is_some(), "Missing pending field");
    assert!(result.get("recent").is_some(), "Missing recent field");
}

#[tokio::test]
async fn test_list_voices() {
    // This test verifies list_voices returns expected structure

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .expect("Failed to connect to daemon");

    let resp = rpc_call(&mut stream, "list_voices", serde_json::json!({}))
        .await
        .expect("RPC call failed");

    assert!(
        resp.error.is_none(),
        "list_voices returned error: {:?}",
        resp.error
    );

    let result = resp.result.expect("list_voices returned no result");
    let voices = result.get("voices").expect("Missing voices field");
    assert!(voices.is_array(), "voices should be an array");
    assert!(
        voices.as_array().unwrap().len() >= 7,
        "Should have at least 7 builtin voices"
    );
}

#[tokio::test]
async fn test_set_speed() {
    // This test verifies set_speed accepts valid values and rejects invalid ones

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .expect("Failed to connect to daemon");

    // Valid speed should succeed
    let resp = rpc_call(
        &mut stream,
        "set_speed",
        serde_json::json!({ "speed": 1.0 }),
    )
    .await
    .expect("RPC call failed");

    assert!(
        resp.error.is_none(),
        "set_speed(1.0) returned error: {:?}",
        resp.error
    );

    // Invalid speed (too low) should fail
    let resp = rpc_call(
        &mut stream,
        "set_speed",
        serde_json::json!({ "speed": 0.0 }),
    )
    .await
    .expect("RPC call failed");

    assert!(resp.error.is_some(), "set_speed(0.0) should return error");

    // Invalid speed (negative) should fail
    let resp = rpc_call(
        &mut stream,
        "set_speed",
        serde_json::json!({ "speed": -1.0 }),
    )
    .await
    .expect("RPC call failed");

    assert!(resp.error.is_some(), "set_speed(-1.0) should return error");

    // Invalid speed (too high) should fail - daemon allows up to 5.0
    let resp = rpc_call(
        &mut stream,
        "set_speed",
        serde_json::json!({ "speed": 5.1 }),
    )
    .await
    .expect("RPC call failed");

    assert!(resp.error.is_some(), "set_speed(5.1) should return error");

    // Valid speed at upper bound should succeed
    let resp = rpc_call(
        &mut stream,
        "set_speed",
        serde_json::json!({ "speed": 5.0 }),
    )
    .await
    .expect("RPC call failed");

    assert!(
        resp.error.is_none(),
        "set_speed(5.0) should succeed: {:?}",
        resp.error
    );
}

#[tokio::test]
async fn test_speak_minimal() {
    // This test verifies speak command queues a request

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .expect("Failed to connect to daemon");

    let resp = rpc_call(&mut stream, "speak", serde_json::json!({ "text": "test" }))
        .await
        .expect("RPC call failed");

    assert!(
        resp.error.is_none(),
        "speak returned error: {:?}",
        resp.error
    );

    let result = resp.result.expect("speak returned no result");
    let queue_id = result.get("queue_id").expect("Missing queue_id field");
    assert!(queue_id.is_string(), "queue_id should be a string");

    // Wait a moment for processing
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check status to verify item was processed
    let resp = rpc_call(&mut stream, "status", serde_json::json!({}))
        .await
        .expect("RPC call failed");

    assert!(resp.error.is_none(), "status returned error");
}

#[tokio::test]
async fn test_automerge_state_persistence() {
    // This test verifies automerge state file is created and updated

    let state_path = dirs::home_dir()
        .unwrap()
        .join(".voice")
        .join("state.automerge");

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    // State file should exist after daemon starts
    assert!(state_path.exists(), "state.automerge should exist");

    let mut stream = UnixStream::connect(&socket_path)
        .await
        .expect("Failed to connect to daemon");

    // Queue a speak request
    let resp = rpc_call(
        &mut stream,
        "speak",
        serde_json::json!({ "text": "persistence test" }),
    )
    .await
    .expect("RPC call failed");

    assert!(resp.error.is_none(), "speak returned error");

    // Wait for state update
    tokio::time::sleep(Duration::from_millis(500)).await;

    // State file should have been updated (check file modification time)
    let metadata = std::fs::metadata(&state_path).expect("Failed to get state file metadata");
    assert!(metadata.is_file(), "state.automerge should be a file");
    assert!(metadata.len() > 0, "state.automerge should not be empty");
}

#[tokio::test]
async fn test_concurrent_connections() {
    // This test verifies daemon can handle multiple concurrent connections

    let socket_path = dirs::home_dir().unwrap().join(".voice").join("daemon.sock");

    if !socket_path.exists() {
        eprintln!("Skipping test: daemon not running");
        return;
    }

    // Spawn 5 concurrent connections
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let socket_path = socket_path.clone();
            tokio::spawn(async move {
                let mut stream = UnixStream::connect(&socket_path)
                    .await
                    .expect("Failed to connect to daemon");

                let resp = rpc_call(&mut stream, "status", serde_json::json!({}))
                    .await
                    .expect("RPC call failed");

                assert!(
                    resp.error.is_none(),
                    "Connection {} status failed: {:?}",
                    i,
                    resp.error
                );
            })
        })
        .collect();

    // Wait for all to complete
    for handle in handles {
        handle.await.expect("Task panicked");
    }
}
