//! Wire protocol for the voice daemon.
//!
//! Defines frame types, RPC request/response types, and codec functions
//! shared between the daemon, MCP server, and CLI client.
//!
//! ## Wire format
//!
//! Length-prefixed frames over a byte stream (Unix socket or pipe):
//!
//! ```text
//! [4 bytes: payload length (big-endian u32)] [1 byte: frame type] [N bytes: payload]
//! ```
//!
//! Frame types:
//! - `0x01` Request  — JSON-RPC 2.0 request
//! - `0x02` Response — JSON-RPC 2.0 response
//! - `0x03` Event    — daemon-initiated notification (state changes, progress)

pub mod client;
pub mod frames;
pub mod rpc;
